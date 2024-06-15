use bevy::core_pipeline::core_2d::Transparent2d;
use bevy::core_pipeline::core_3d::Transparent3d;
use bevy::core_pipeline::tonemapping::{
    get_lut_bind_group_layout_entries, DebandDither, Tonemapping,
};
use bevy::ecs::entity::EntityHashMap;
use bevy::ecs::query::QueryItem;
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
    BindGroupLayout, BindGroupLayoutEntries, BlendState, Buffer, BufferInitDescriptor,
    BufferUsages, CachedRenderPipelineId, ColorTargetState, ColorWrites, FragmentState, FrontFace,
    ImageDataLayout, IndexFormat, PipelineCache, PolygonMode, PrimitiveState, RenderPassDescriptor,
    RenderPipelineDescriptor, Sampler, SamplerBindingType, ShaderStages, SpecializedMeshPipeline,
    SpecializedMeshPipelineError, SpecializedMeshPipelines, SpecializedRenderPipeline,
    SpecializedRenderPipelines, TextureFormat, TextureSampleType, TextureViewDescriptor,
    VertexAttribute, VertexBufferLayout, VertexFormat, VertexState, VertexStepMode,
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

pub(crate) fn plugin(app: &mut App) {
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        warn!("Failed to get render app for tilemap_renderer");
        return;
    };

    render_app.init_resource::<SpecializedRenderPipelines<TileMapPipeline>>();
    render_app.add_systems(ExtractSchedule, extract_tilemap_chunks);
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
        // TODO: maybe be smarter about this, don't hardcode 2048x2048
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

/// Minimal representation needed for rendering.
pub struct ExtractedTileMap {
    pub transform: GlobalTransform,
    /// Select an area of the texture
    pub chunk: TilemapChunk,
    /// For cases where additional [`ExtractedTileMaps`] are created during extraction, this stores the
    /// entity that caused that creation for use in determining visibility.
    pub original_entity: Option<Entity>,
}

// Not sure if we want to use a hashmap.
// the benefit for bevy_sprite was that it could be cleared, so you get "free" destruction detection.
// But we want to preserve maps.
#[derive(Resource, Default)]
pub struct ExtractedTileMaps {
    pub map: EntityHashMap<ExtractedTileMap>,
}

// A temporary implementation that just initializes placeholders onto the gpu
pub fn extract_tilemap_chunks(
    mut commands: Commands,
    mut extracted_tilemap: ResMut<ExtractedTileMaps>,
    // todo: figure out how to get Tile Texture data, and how it should be formatted
    //  we probably want to put that into a different extraction function.
    //  since it should change very rarely.
    tilemap_query: Extract<Query<(Entity, &ViewVisibility, &TilemapChunks, &GlobalTransform)>>,
) {
    for (entity, view_visibility, chunk, transform) in tilemap_query.iter() {
        if !view_visibility.get() {
            continue;
        }

        match extracted_tilemap.map.entry(entity) {
            Entry::Occupied(o_map) => {
                // Todo: we can transfer all "dirty" parts of the chunk here.
            }
            Entry::Vacant(v_map) => {
                // otherwise we just copy the whole thing, since its the first time.
                v_map.insert(ExtractedTileMap {
                    transform: *transform,
                    // todo: decide how to handle multiple chunks.
                    chunk: chunk.chunks[0].clone(),
                    original_entity: None,
                });
            }
        };
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

struct TileMapPipeline {
    tilemap_layout: BindGroupLayout,
    tiles_layout: BindGroupLayout,
    //pipeline_id: CachedRenderPipelineId,
    vertex_shader: Handle<Shader>,
    fragment_shader: Handle<Shader>,
}

// Initialize the pipelines data
impl FromWorld for TileMapPipeline {
    fn from_world(world: &mut World) -> Self {
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
            vertex_shader: Default::default(),
            fragment_shader: Default::default(),
        }
    }
}

// Specialize the pipeline with size/runtime configurable data. I think.
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
                shader: self.vertex_shader.clone(),
                shader_defs: vec![],
                entry_point: "vertex".into(),
                buffers: vec![vertex_buffer_layout],
            },
            fragment: Some(FragmentState {
                shader: self.fragment_shader.clone(),
                shader_defs: vec![],
                entry_point: Default::default(),
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
    mut views: Query<(
        Entity,
        &VisibleEntities,
        &ExtractedView,
        Option<&Tonemapping>,
        Option<&DebandDither>,
    )>,
) {
    let draw_tilemaps_function = draw_functions.read().id::<DrawSprite>();

    for (view_entity, visible_entities, view, tonemapping, dither) in &mut views {
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view_entity) else {
            continue;
        };

        transparent_phase
            .items
            .reserve(extracted_tilemaps.map.len());
        // Todo: finish fleshing out this function.
        for (entity, extracted_tilemap) in extracted_tilemaps.map.iter() {
            let index = extracted_sprite.original_entity.unwrap_or(*entity).index();

            if !view_entities.contains(index as usize) {
                continue;
            }

            // These items will be sorted by depth with other phase items
            let sort_key = FloatOrd(extracted_tilemap.transform.translation().z);

            // Add the item to the render phase
            transparent_phase.add(Transparent2d {
                draw_function: draw_sprite_function,
                pipeline,
                entity: *entity,
                sort_key,
                // batch_range and dynamic_offset will be calculated in prepare_sprites
                batch_range: 0..0,
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
    // SetTilemapViewBindGroup<0>,
    //SetTilemapTextureBindGroup<1>,
    //SetTilesTextureBindGroup<2>,
    DrawTileMap,
);

/*pub struct SetTilemapViewBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetTilemapViewBindGroup<I> {
    type Param = ();
    type ViewQuery = (Read<ViewUniformOffset>, Read<SpriteViewBindGroup>);
    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        (view_uniform, tilemap_view_bind_group): ROQueryItem<'w, Self::ViewQuery>,
        _entity: Option<()>,
        _param: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_bind_group(I, &sprite_view_bind_group.value, &[view_uniform.offset]);
        RenderCommandResult::Success
    }
}

pub struct SetTilemapTextureBindGroup<const I: usize>;
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
}

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
    type Param = SRes<ImageBindGroups>;
    type ViewQuery = ();
    type ItemQuery = Read<TilemapChunk>;

    fn render<'w>(
        _item: &P,
        _view: (),
        batch: Option<&'_ TilemapChunk>,
        sprite_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let sprite_meta = sprite_meta.into_inner();
        let Some(batch) = batch else {
            return RenderCommandResult::Failure;
        };
        let tilemap_size = 2048;
        pass.draw(0..6 * tilemap_size, 0..1);
        RenderCommandResult::Success
    }
}
