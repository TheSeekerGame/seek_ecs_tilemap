use bevy::core_pipeline::core_2d::Transparent2d;
use bevy::core_pipeline::core_3d::Transparent3d;
use bevy::core_pipeline::tonemapping::{
    get_lut_bind_group_layout_entries, DebandDither, Tonemapping,
};
use bevy::ecs::entity::EntityHashMap;
use bevy::ecs::query::{QueryItem, ROQueryItem};
use bevy::ecs::system::lifetimeless::{Read, SRes};
use bevy::ecs::system::{SystemParamItem, SystemState};
use bevy::math::{Affine3, FloatOrd};
use bevy::prelude::*;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::mesh::{GpuBufferInfo, GpuMesh, MeshVertexBufferLayoutRef, PrimitiveTopology};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo};
use bevy::render::render_phase::{
    AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
    RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
};
use bevy::render::render_resource::binding_types::{sampler, texture_2d, uniform_buffer};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::texture::{
    BevyDefault, DefaultImageSampler, GpuImage, ImageSampler, TextureFormatPixelInfo,
};
use bevy::render::view::{
    ExtractedView, NoFrustumCulling, ViewUniform, ViewUniformOffset, ViewUniforms, VisibleEntities
};
use bevy::render::{Extract, Render, RenderApp, RenderSet};
use bevy::utils::hashbrown::hash_map::Entry;
use binding_types::texture_2d_array;
use std::borrow::Cow;

use crate::{map::*, tiles::*};

pub struct TileMapRendererPlugin;
impl Plugin for TileMapRendererPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PostUpdate, update_tilemap_chunks);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            warn!("Failed to get render app for tilemap_renderer");
            return;
        };

        render_app
            .init_resource::<SpecializedRenderPipelines<TilemapPipeline>>()
            .init_resource::<ExtractedTilemaps>()
            .init_resource::<PreparedTilemaps>()
            .add_render_command::<Transparent2d, DrawTilemap>()
            .add_systems(ExtractSchedule, extract_tilemaps)
            .add_systems(Render, (
                prepare_tilemaps.in_set(RenderSet::Prepare),
                queue_tilemaps.in_set(RenderSet::Queue),
            ));
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<TilemapPipeline>();
        }
    }
}

/// GPU representation of TilemapChunks
struct GpuTilemapChunks {
    texture: Texture,
    texture_view: TextureView,
}

/// Minimal representation needed for rendering.
struct ExtractedTilemap {
    transform: GlobalTransform,
    chunks: TilemapChunks,
    tile_size: TilemapTileSize,
    grid_size: TilemapGridSize,
}

#[derive(Resource, Default)]
struct ExtractedTilemaps {
    map: EntityHashMap<ExtractedTilemap>,
}

struct GpuTilemap {
    gpu_chunks: GpuTilemapChunks,
    tilemap_uniform: UniformBuffer<TilemapInfo>,
    view_bind_group: BindGroup,
    tilemap_bind_group: BindGroup,
    tile_bind_group: Option<BindGroup>,
}

#[derive(Resource, Default)]
pub struct PreparedTilemaps {
    map: EntityHashMap<GpuTilemap>,
}

const SHADER_ASSET_PATH: &str = "tile_map_render.wgsl";

#[derive(ShaderType, Clone)]
struct TilemapInfo {
    transform: [Vec4; 3],
    tile_size: Vec2,
    grid_size: Vec2,
    n_tiles_per_chunk: UVec2,
    n_chunks: UVec2,
}

#[derive(Resource)]
struct TilemapPipeline {
    view_layout: BindGroupLayout,
    tilemap_layout: BindGroupLayout,
    tiles_layout: BindGroupLayout,
    shader: Handle<Shader>,
}

// Initialize the pipelines data
impl FromWorld for TilemapPipeline {
    fn from_world(world: &mut World) -> Self {
        let shader = world.load_asset(SHADER_ASSET_PATH);
        let mut system_state: SystemState<(
            Res<RenderDevice>,
            Res<DefaultImageSampler>,
            Res<RenderQueue>,
        )> = SystemState::new(world);
        let (render_device, default_sampler, render_queue) = system_state.get_mut(world);

        let view_layout = render_device.create_bind_group_layout(
            "tilemap_view_layout",
            &BindGroupLayoutEntries::single(
                ShaderStages::VERTEX_FRAGMENT,
                uniform_buffer::<ViewUniform>(true),
            ),
        );
        let tilemap_layout = render_device.create_bind_group_layout(
            "tilemap_data_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX_FRAGMENT,
                (
                    uniform_buffer::<TilemapInfo>(false),
                    texture_2d_array(TextureSampleType::Uint),
                ),
            ),
        );
        let tiles_layout = render_device.create_bind_group_layout(
            "tilemap_tiledata_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    sampler(SamplerBindingType::Filtering),
                ),
            ),
        );

        TilemapPipeline {
            view_layout,
            tilemap_layout,
            tiles_layout,
            shader,
        }
    }
}

// Specialize the pipeline with size/runtime configurable data. I think.
#[derive(Clone, Hash, PartialEq, Eq)]
pub struct TilemapPipelineKey {
    // TODO: tile texture
    msaa_samples: u32,
}

impl SpecializedRenderPipeline for TilemapPipeline {
    type Key = TilemapPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("tilemap_pipeline".into()),
            layout: vec![self.view_layout.clone(), self.tilemap_layout.clone()], // TODO: tile texture
            push_constant_ranges: vec![],
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: vec![], // TODO: tile texture
                entry_point: "vertex".into(),
                buffers: vec![],
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    // todo: make this support HDR at some point?
                    format: TextureFormat::bevy_default(),
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: key.msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        }
    }
}

fn update_tilemap_chunks(
    mut q_map: Query<(&mut TilemapChunks, &TilemapSize)>,
    q_tile: Query<
        (
            &TilemapId,
            &TilePos,
            Ref<TileTextureIndex>,
            Ref<TileColor>,
            Ref<TileFlip>,
            Ref<TileVisible>,
        ),
    >,
) {
    // first, init things if necessary
    for (mut chunks, size) in &mut q_map {
        if !chunks.is_added() {
            continue;
        }
        // TODO: don't hardcode 2048x2048, this can be optimized
        // if map size is >2048, divide into roughly equal chunks
        // that have minimal wastage when padded to 64x64
        chunks.init((*size).into(), UVec2::new(2048, 2048));
    }
    // now, update any changed tiles
    // memoize tilemap lookup for perf
    let mut last_map_id = None;
    let mut last_map = None;
    for (tid, pos, index, color, flip, vis) in &q_tile {
        if last_map_id != Some(*tid) {
            last_map_id = None;
            last_map = None;
            let Ok(map) = q_map.get_mut(tid.0) else {
                continue;
            };
            last_map_id = Some(*tid);
            last_map = Some(map);
        }
        let Some((ref mut chunks, _)) = last_map else {
            unreachable!()
        };
        let tile_changed = index.is_changed() || color.is_changed() || flip.is_changed() || vis.is_changed();
        if chunks.is_added() || tile_changed {
            chunks.set_tiledata_at(pos, &*index, &*color, &*flip, &*vis);
        }
    }
}

fn extract_tilemaps(
    mut extracted_tilemaps: ResMut<ExtractedTilemaps>,
    tilemap_query: Extract<Query<(
        Entity,
        &ViewVisibility,
        &GlobalTransform,
        &TilemapChunks,
        &TilemapTileSize,
        &TilemapGridSize,
    )>>,
) {
    for (entity, view_visibility, transform, chunks, tile_size, grid_size) in tilemap_query.iter() {
        // TODO: in order for this to actually work, we need a system in the
        // main world that knows how to do frustum culling for tilemaps
        if !view_visibility.get() {
            // continue;
        }

        match extracted_tilemaps.map.entry(entity) {
            Entry::Occupied(mut o_map) => {
                // Transfer all "dirty" parts of the chunks here.
                let map = o_map.get_mut();
                map.transform = transform.clone();
                map.chunks.copy_dirty(chunks);
            }
            Entry::Vacant(v_map) => {
                // otherwise copy all chunks, since it's the first time.
                v_map.insert(ExtractedTilemap {
                    transform: transform.clone(),
                    chunks: chunks.clone(),
                    tile_size: *tile_size,
                    grid_size: *grid_size,
                });
            }
        };
    }
}

fn prepare_tilemaps(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut extracted_tilemaps: ResMut<ExtractedTilemaps>,
    mut prepared_tilemaps: ResMut<PreparedTilemaps>,
    tilemap_pipeline: Res<TilemapPipeline>,
    view_uniforms: Res<ViewUniforms>,
) {
    for (e, extracted) in extracted_tilemaps.map.iter_mut() {
        if let Some(prepared) = prepared_tilemaps.map.get_mut(e) {
            // Texture already exists in GPU memory.
            // Update it with any dirty data!
            prepared.gpu_chunks.copy_dirty(&queue, &extracted.chunks);
            // Tilemap Uniform already exists,
            // but we need to update the data in the buffer.
            prepared.tilemap_uniform.set(TilemapInfo {
                transform: Affine3::from(&extracted.transform.affine()).to_transpose(),
                tile_size: extracted.tile_size.into(),
                grid_size: extracted.grid_size.into(),
                n_tiles_per_chunk: extracted.chunks.chunk_size,
                n_chunks: extracted.chunks.n_chunks,
            });
            prepared.tilemap_uniform.write_buffer(&device, &queue);
            // Bind Groups already exist and don't need changing.
        } else {
            let Some(view_binding) = view_uniforms.uniforms.binding() else {
                continue;
            };
            // Setup the GPU texture.
            let gpu_chunks = GpuTilemapChunks::new(&device, &extracted.chunks);
            gpu_chunks.copy_all(&queue, &extracted.chunks);
            extracted.chunks.clear_all_dirty_bitmaps();
            // Setup the tilemap uniform
            let tilemap_info = TilemapInfo {
                transform: Affine3::from(&extracted.transform.affine()).to_transpose(),
                tile_size: extracted.tile_size.into(),
                grid_size: extracted.grid_size.into(),
                n_tiles_per_chunk: extracted.chunks.chunk_size,
                n_chunks: extracted.chunks.n_chunks,
            };
            let mut tilemap_uniform = UniformBuffer::from(tilemap_info);
            tilemap_uniform.set_label(Some("tilemap_uniform"));
            tilemap_uniform.write_buffer(&device, &queue);
            // Setup the bind groups
            let view_bind_group = device.create_bind_group(
                "tilemap_view_bind_group",
                &tilemap_pipeline.view_layout,
                &BindGroupEntries::single(view_binding),
            );
            let tilemap_bind_group = device.create_bind_group(
                "tilemap_bind_group",
                &tilemap_pipeline.tilemap_layout,
                &BindGroupEntries::sequential((
                    &tilemap_uniform,
                    &gpu_chunks.texture_view,
                )),
            );
            prepared_tilemaps.map.insert(*e, GpuTilemap {
                gpu_chunks,
                tilemap_uniform,
                view_bind_group,
                tilemap_bind_group,
                tile_bind_group: None, // TODO: tile texture
            });
        }
    }
}

fn queue_tilemaps(
    draw_functions: Res<DrawFunctions<Transparent2d>>,
    extracted_tilemaps: Res<ExtractedTilemaps>,
    tilemap_pipeline: Res<TilemapPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<TilemapPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    msaa: Res<Msaa>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent2d>>,
    mut views: Query<(
        Entity,
        &VisibleEntities,
        &ExtractedView,
        Option<&Tonemapping>,
        Option<&DebandDither>,
    )>,
) {
    let draw_tilemap_function = draw_functions.read().id::<DrawTilemap>();

    for (view_entity, visible_entities, view, tonemapping, dither) in &mut views {
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view_entity) else {
            continue;
        };

        let pipeline = pipelines.specialize(
            &pipeline_cache,
            &tilemap_pipeline,
            TilemapPipelineKey {
                msaa_samples: msaa.samples(),
            },
        );

        transparent_phase
            .items
            .reserve(extracted_tilemaps.map.len());

        for (entity, extracted_tilemap) in extracted_tilemaps.map.iter() {
            /*let index = extracted_tilemap.unwrap_or(*entity).index();

            if !view_entities.contains(index as usize) {
                continue;
            }*/

            // These items will be sorted by depth with other phase items
            let sort_key = FloatOrd(extracted_tilemap.transform.translation().z);

            // Add the item to the render phase
            transparent_phase.add(Transparent2d {
                draw_function: draw_tilemap_function,
                pipeline,
                entity: *entity,
                sort_key,
                // I think this needs to be at least 1
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::NONE,
            });
        }
    }
}

impl GpuTilemapChunks {
    fn new(device: &RenderDevice, chunks: &TilemapChunks) -> Self {
        let desc_texture = TextureDescriptor {
            label: Some("seek_ecs_tilemap_chunks"),
            size: chunks.texture_size(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Uint,
            usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc_texture);
        let desc_view = TextureViewDescriptor {
            label: Some("seek_ecs_tilemap_chunks"),
            format: Some(TextureFormat::Rgba32Uint),
            dimension: Some(TextureViewDimension::D2Array),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        };
        let texture_view = texture.create_view(&desc_view);
        Self {
            texture,
            texture_view,
        }
    }
    fn copy_all(&self, queue: &RenderQueue, chunks: &TilemapChunks) {
        for (z, chunk) in chunks.chunks.iter().enumerate() {
            let mut data_start = 0;
            for sc_y in (0..32).take(chunks.n_subchunks.y as usize) {
                for sc_x in (0..32).take(chunks.n_subchunks.x as usize) {
                    let data_end = data_start + TilemapChunks::SUBCHUNK_DATA_LEN;
                    let data = &chunk.data[data_start..data_end];
                    self.copy_subchunk_data(queue, sc_x, sc_y, z as u32, data);
                    data_start = data_end;
                }
            }
        }
    }
    fn copy_dirty(&self, queue: &RenderQueue, chunks: &TilemapChunks) {
        for (z, chunk) in chunks.chunks.iter().enumerate() {
            let mut data_start = 0;
            for (sc_y, row_bitmap) in chunk
                .dirty_bitmap
                .iter()
                .copied()
                .take(chunks.n_subchunks.y as usize)
                .enumerate()
            {
                for sc_x in (0..32).take(chunks.n_subchunks.x as usize) {
                    let data_end = data_start + TilemapChunks::SUBCHUNK_DATA_LEN;
                    if row_bitmap & (1 << sc_x) != 0 {
                        let data = &chunk.data[data_start..data_end];
                        self.copy_subchunk_data(queue, sc_x, sc_y as u32, z as u32, data);
                    }
                    data_start = data_end;
                }
            }
        }
    }
    fn copy_subchunk_data(&self, queue: &RenderQueue, x: u32, y: u32, z: u32, data: &[u8]) {
        let texture = ImageCopyTexture {
            texture: &self.texture,
            mip_level: 0,
            origin: Origin3d {
                x: x * 64,
                y: y * 64,
                z,
            },
            aspect: TextureAspect::All,
        };
        let data_layout = ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(64 * 16),
            rows_per_image: Some(64),
        };
        let size = Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        };
        queue.write_texture(texture, data, data_layout, size);
    }
}

/// [`RenderCommand`]s for TileMap rendering.
type DrawTilemap = (
    SetItemPipeline,
    SetTilemapViewBindGroup<0>,
    SetTilemapBindGroup<1>,
    //SetTilesTextureBindGroup<2>,
    DrawTileMap,
);

pub struct SetTilemapViewBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetTilemapViewBindGroup<I> {
    type Param = SRes<PreparedTilemaps>;
    type ViewQuery = Read<ViewUniformOffset>;
    type ItemQuery = ();

    fn render<'w>(
        item: &P,
        view_uniform: ROQueryItem<'w, Self::ViewQuery>,
        _entity: Option<()>,
        tilemaps: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let tilemaps = tilemaps.into_inner();
        let Some(tilemap) = tilemaps.map.get(&item.entity()) else {
            return RenderCommandResult::Failure;
        };
        pass.set_bind_group(I, &tilemap.view_bind_group, &[view_uniform.offset]);
        RenderCommandResult::Success
    }
}

pub struct SetTilemapBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetTilemapBindGroup<I> {
    type Param = SRes<PreparedTilemaps>;
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        item: &P,
        _view: (),
        _entity: Option<()>,
        tilemaps: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let tilemaps = tilemaps.into_inner();
        let Some(tilemap) = tilemaps.map.get(&item.entity()) else {
            return RenderCommandResult::Failure;
        };
        pass.set_bind_group(I, &tilemap.tilemap_bind_group, &[]);
        RenderCommandResult::Success
    }
}
/*
pub struct SetTilesTextureBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetTilesTextureBindGroup<I> {
    type Param = SRes<ImageBindGroups>;
    type ViewQuery = ();
    type ItemQuery = Read<TilemapChunk>;

    fn render<'w>(
        _item: &P,
        _view: (),
        batch: Option<&'_ TilemapChunk>,
        image_bind_groups: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let image_bind_groups = image_bind_groups.into_inner();
        let Some(batch) = batch else {
            return RenderCommandResult::Failure;
        };

        pass.set_bind_group(
            I,
            image_bind_groups
                .values
                .get(&batch.image_handle_id)
                .unwrap(),
            &[],
        );
        RenderCommandResult::Success
    }
}*/

struct DrawTileMap {}
impl<P: PhaseItem> RenderCommand<P> for DrawTileMap {
    type Param = SRes<ExtractedTilemaps>;
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        item: &P,
        _view: (),
        _query: Option<()>,
        tilemaps: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let tilemaps = tilemaps.into_inner();
        let Some(tilemap) = tilemaps.map.get(&item.entity()) else {
            return RenderCommandResult::Failure;
        };
        let n_verts = tilemap.chunks.chunk_size.element_product() * 6;
        let n_insts = tilemap.chunks.n_chunks.element_product();
        pass.draw(0..n_verts, 0..n_insts);
        RenderCommandResult::Success
    }
}
