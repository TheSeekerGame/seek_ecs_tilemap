use bevy::core_pipeline::core_2d::Transparent2d;
use bevy::core_pipeline::core_3d::Transparent3d;
use bevy::core_pipeline::tonemapping::{
    get_lut_bind_group_layout_entries, DebandDither, Tonemapping,
};
use bevy::ecs::entity::EntityHashMap;
use bevy::ecs::query::{QueryItem, ROQueryItem};
use bevy::ecs::system::lifetimeless::{Read, SRes};
use bevy::ecs::system::{SystemParamItem, SystemState};
use bevy::math::FloatOrd;
use bevy::pbr::{
    MeshPipeline, MeshPipelineKey, RenderMeshInstances, SetMeshBindGroup, SetMeshViewBindGroup,
};
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
use bevy::render::render_resource::{
    BindGroup, BindGroupLayout, BindGroupLayoutEntries, BlendState, Buffer, BufferInitDescriptor,
    BufferUsages, CachedRenderPipelineId, ColorTargetState, ColorWrites, Extent3d, FragmentState,
    FrontFace, ImageDataLayout, IndexFormat, PipelineCache, PolygonMode, PrimitiveState,
    RenderPassDescriptor, RenderPipelineDescriptor, Sampler, SamplerBindingType, ShaderStages,
    SpecializedMeshPipeline, SpecializedMeshPipelineError, SpecializedMeshPipelines,
    SpecializedRenderPipeline, SpecializedRenderPipelines, Texture, TextureAspect,
    TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
    TextureView, TextureViewDescriptor, TextureViewDimension, VertexAttribute, VertexBufferLayout,
    VertexFormat, VertexState, VertexStepMode,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::texture::{
    BevyDefault, DefaultImageSampler, GpuImage, ImageSampler, TextureFormatPixelInfo,
};
use bevy::render::view::{
    ExtractedView, NoFrustumCulling, ViewUniform, ViewUniformOffset, VisibleEntities,
};
use bevy::render::{Extract, Render, RenderApp, RenderSet};
use bevy::sprite::{
    DrawSprite, DrawSpriteBatch, ExtractedSprite, ExtractedSprites, ImageBindGroups,
    SetSpriteTextureBindGroup, SetSpriteViewBindGroup, SpritePipeline,
};
use bevy::utils::hashbrown::hash_map::Entry;
use std::borrow::Cow;

use crate::{map::*, tiles::*};

pub struct TileMapRendererPlugin;
impl Plugin for TileMapRendererPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            warn!("Failed to get render app for tilemap_renderer");
            return;
        };

        render_app
            .init_resource::<SpecializedRenderPipelines<TileMapPipeline>>()
            .init_resource::<ExtractedTileMaps>()
            .add_render_command::<Transparent2d, DrawTilemap>()
            .add_systems(ExtractSchedule, extract_tilemaps)
            .add_systems(
                Render,
                prepare_tilemap_chunks.in_set(RenderSet::PrepareResources),
            )
            .add_systems(Render, queue_tilemaps.in_set(RenderSet::Queue));
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<TileMapPipeline>();
        }
    }
}

/// GPU representation of TilemapChunks
struct GpuTilemapChunks {
    texture: Texture,
    texture_view: TextureView,
}

/// Minimal representation needed for rendering.
pub struct ExtractedTileMap {
    pub transform: GlobalTransform,
    pub chunks: TilemapChunks,
    pub gpu_chunks: Option<GpuTilemapChunks>,
}

#[derive(Resource, Default)]
pub struct ExtractedTileMaps {
    pub map: EntityHashMap<ExtractedTileMap>,
}

impl GpuTilemapChunks {
    fn new(device: &RenderDevice, chunks: &TilemapChunks) -> Self {
        let desc_texture = TextureDescriptor {
            label: Some("seek_ecs_tilemap_chunks"),
            size: Extent3d {
                width: 1000,
                height: 1000,
                depth_or_array_layers: 4,
            }, //chunks.texture_size(),
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
    fn copy_all(&mut self, queue: &RenderQueue, chunks: &TilemapChunks) {
        for chunk in chunks.chunks.iter() {
            let mut data_start = 0;
            for sc_y in (0..32).take(chunks.n_subchunks.y as usize) {
                for sc_x in (0..32).take(chunks.n_subchunks.y as usize) {
                    let data_end = data_start + TilemapChunks::SUBCHUNK_DATA_LEN;
                    let data = &chunk.data[data_start..data_end];
                    // TODO: queue.write_texture(texture, data, data_layout, size);
                    data_start = data_end;
                }
            }
        }
    }
    fn copy_dirty(&mut self, queue: &RenderQueue, chunks: &TilemapChunks) {
        for chunk in chunks.chunks.iter() {
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
                        // TODO: queue.write_texture(texture, data, data_layout, size);
                    }
                    data_start = data_end;
                }
            }
        }
    }
}

// This preprocess all the tiles on the main world, running the expensive change detection there
// to give us a compressed set of changes, the tilemapchunks

// These are then processed at some point and minimal data is sent from them to the render world

// The render world then uploads this data to the gpu.

fn update_tilemap_chunks(
    mut q_map: Query<(&mut TilemapChunks, &TilemapSize)>,
    q_tile: Query<
        (
            &TilemapId,
            &TilePos,
            &TileTextureIndex,
            &TileColor,
            &TileFlip,
            &TileVisible,
        ),
        Or<(
            Changed<TileTextureIndex>,
            Changed<TileColor>,
            Changed<TileFlip>,
            Changed<TileVisible>,
        )>,
    >,
) {
    // first, init things if necessary
    for (mut chunks, size) in &mut q_map {
        if !chunks.is_added() {
            continue;
        }
        todo!();
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
        if chunks.is_added() {
            // If just added, we have already set everything above.
            continue;
        }
        chunks.set_tiledata_at(pos, index, color, flip, vis);
    }
}

// A temporary implementation that just initializes placeholders onto the gpu
pub fn extract_tilemaps(
    mut extracted_tilemaps: ResMut<ExtractedTileMaps>,
    // todo: figure out how to get Tile Texture data, and how it should be formatted
    //  we probably want to put that into a different extraction function.
    //  since it should change very rarely.
    tilemap_query: Extract<Query<(Entity, &ViewVisibility, &TilemapChunks, &GlobalTransform)>>,
) {
    for (entity, view_visibility, chunks, transform) in tilemap_query.iter() {
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
                v_map.insert(ExtractedTileMap {
                    transform: transform.clone(),
                    chunks: chunks.clone(),
                    gpu_chunks: None, // This will be handled in Prepare
                });
            }
        };
    }
}

fn prepare_tilemap_chunks(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut extracted_tilemaps: ResMut<ExtractedTileMaps>,
) {
    //println!("prepared: {}", extracted_tilemaps.map.len());
    for map in extracted_tilemaps.map.values_mut() {
        if let Some(gpu_chunks) = &mut map.gpu_chunks {
            // Texture already exists in GPU memory.
            // Update it with any dirty data!
            gpu_chunks.copy_dirty(&queue, &map.chunks);
        } else {
            // First run; setup the GPU texture.
            let mut gpu_chunks = GpuTilemapChunks::new(&device, &map.chunks);
            gpu_chunks.copy_all(&queue, &map.chunks);
            map.chunks.clear_all_dirty_bitmaps();
            map.gpu_chunks = Some(gpu_chunks);
        }
    }
}

// Todo: rewaork taking into account that we are storing the TileMap data as texture arraays
//  not just a single texture.
// For the pipeline, we can do very similar to the bevy_sprite pipeline.
// main differences are we have two textures, one storing the tile *map*
// and the other storing the tiles textures.

// The vertex shader will use vertex pulling from the tilemap_data to calculate vertex positions
// but we will still need to initialize the vertex buffer to the correct size.

// (Also, we don't have to do batching or anything like that; since the tile map
// structure batches tiles naturally)

const SHADER_ASSET_PATH: &str = "tile_map_render.wgsl";

#[derive(Resource)]
struct TileMapPipeline {
    tilemap_layout: BindGroupLayout,
    tiles_layout: BindGroupLayout,
    //pipeline_id: CachedRenderPipelineId,
    shader: Handle<Shader>,
}

// Initialize the pipelines data
impl FromWorld for TileMapPipeline {
    fn from_world(world: &mut World) -> Self {
        let shader = world.load_asset(SHADER_ASSET_PATH);
        let mut system_state: SystemState<(
            Res<RenderDevice>,
            Res<DefaultImageSampler>,
            Res<RenderQueue>,
        )> = SystemState::new(world);
        let (render_device, default_sampler, render_queue) = system_state.get_mut(world);

        // Each tilemap needs two materials:
        // 1. a large texture that represents the tilemap data
        // (where we store per tile information.)
        // 2. Detailed textures (ie the textures displayed on each individual tile)
        // could probably be renamed.
        let tilemap_layout = render_device.create_bind_group_layout(
            "tilemap_data",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX,
                (texture_2d(TextureSampleType::Uint),),
            ),
        );
        let tiles_layout = render_device.create_bind_group_layout(
            "tilemap_tiles",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    sampler(SamplerBindingType::Filtering),
                ),
            ),
        );

        // Not sure if this is needed or not; might need it as a deafult/empty texture?
        let dummy_white_gpu_image = {
            let image = Image::default();
            let texture = render_device.create_texture(&image.texture_descriptor);
            let sampler = match image.sampler {
                ImageSampler::Default => (**default_sampler).clone(),
                ImageSampler::Descriptor(ref descriptor) => {
                    render_device.create_sampler(&descriptor.as_wgpu())
                }
            };

            let format_size = image.texture_descriptor.format.pixel_size();
            render_queue.write_texture(
                texture.as_image_copy(),
                &image.data,
                ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(image.width() * format_size as u32),
                    rows_per_image: None,
                },
                image.texture_descriptor.size,
            );
            let texture_view = texture.create_view(&TextureViewDescriptor::default());
            GpuImage {
                texture,
                texture_view,
                texture_format: image.texture_descriptor.format,
                sampler,
                size: image.size(),
                mip_level_count: image.texture_descriptor.mip_level_count,
            }
        };

        TileMapPipeline {
            tilemap_layout,
            tiles_layout,
            //pipeline_id: (),
            // todo: need to create the shader and stick it here soon
            shader,
        }
    }
}

// Specialize the pipeline with size/runtime configurable data. I think.
#[derive(Clone, Hash, PartialEq, Eq)]
pub struct TileMapPipelineKey;
impl SpecializedRenderPipeline for TileMapPipeline {
    type Key = TileMapPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        // Don't need key, since

        // We still need a vertex buffer, even though we will not be writing to it from the cpu

        // But we only need position data; *maybe* we also need "uv" data to pass to the frag
        // shader/be interpolated. this will also be generated on the gpu though.
        // The data is 2d; so might be able to share within a single vec4?
        // That should still get interpolated properly
        let vertex_buffer_layout = VertexBufferLayout {
            array_stride: 16,
            step_mode: VertexStepMode::Instance,
            attributes: vec![
                // @location(0) position: vec4<f32>,
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 0,
                },
            ],
        };

        RenderPipelineDescriptor {
            label: Some("TileMapPipeline".into()),
            layout: vec![self.tilemap_layout.clone(), self.tiles_layout.clone()],
            push_constant_ranges: vec![],
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: "vertex".into(),
                buffers: vec![vertex_buffer_layout],
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
            multisample: Default::default(),
        }
    }
}

fn queue_tilemaps(
    draw_functions: Res<DrawFunctions<Transparent2d>>,
    extracted_tilemaps: Res<ExtractedTileMaps>,
    tilemap_pipeline: Res<TileMapPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<TileMapPipeline>>,
    pipeline_cache: Res<PipelineCache>,
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

        let pipeline = pipelines.specialize(&pipeline_cache, &tilemap_pipeline, TileMapPipelineKey);

        transparent_phase
            .items
            .reserve(extracted_tilemaps.map.len());

        // Todo: finish fleshing out this function.
        //println!("queued: {}", extracted_tilemaps.map.len());
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

// Take data from the render world, and parse it into gpu usuable data.
// for now we just throw some test data at it
fn prepare_tilemap_image_bind_groups() {}

// Render everything.

/// [`RenderCommand`]s for TileMap rendering.
pub type DrawTilemap = (
    SetItemPipeline,
    SetTilemapViewBindGroup<0>,
    //SetTilemapTextureBindGroup<1>,
    //SetTilesTextureBindGroup<2>,
    DrawTileMap,
);

#[derive(Component)]
pub struct TilemapViewBindGroup {
    pub value: BindGroup,
}

pub struct SetTilemapViewBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetTilemapViewBindGroup<I> {
    type Param = ();
    type ViewQuery = (Read<ViewUniformOffset>, Read<TilemapViewBindGroup>);
    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        (view_uniform, tilemap_view_bind_group): ROQueryItem<'w, Self::ViewQuery>,
        _entity: Option<()>,
        _param: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_bind_group(I, &tilemap_view_bind_group.value, &[view_uniform.offset]);
        RenderCommandResult::Success
    }
}

// Todo: setup the TilemapTextureArrayBindGroup
/*pub struct SetTilemapTextureBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetTilemapTextureBindGroup<I> {
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
    type Param = SRes<ExtractedTileMaps>;
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        item: &P,
        _view: (),
        _query: Option<()>,
        maps: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(tile_map) = maps.map.get(&item.entity()) else {
            return RenderCommandResult::Failure;
        };
        println!("drawing: {}", &item.entity());
        pass.draw(0..6 * tile_map.chunks.chunk_size.element_product(), 0..1);
        RenderCommandResult::Success
    }
}
