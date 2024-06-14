use bevy::core_pipeline::core_2d::Transparent2d;
use bevy::core_pipeline::core_3d::Transparent3d;
use bevy::core_pipeline::tonemapping::get_lut_bind_group_layout_entries;
use bevy::ecs::query::QueryItem;
use bevy::ecs::system::lifetimeless::{Read, SRes};
use bevy::ecs::system::{SystemParamItem, SystemState};
use bevy::pbr::{
    MeshPipeline, MeshPipelineKey, RenderMeshInstances, SetMeshBindGroup, SetMeshViewBindGroup,
};
use bevy::prelude::*;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::mesh::{GpuBufferInfo, GpuMesh, MeshVertexBufferLayoutRef};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo};
use bevy::render::render_phase::{
    AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
    RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
};
use bevy::render::render_resource::binding_types::{sampler, texture_2d, uniform_buffer};
use bevy::render::render_resource::{
    BindGroupLayout, BindGroupLayoutEntries, Buffer, BufferInitDescriptor, BufferUsages,
    CachedRenderPipelineId, FragmentState, ImageDataLayout, IndexFormat, PipelineCache,
    RenderPassDescriptor, RenderPipelineDescriptor, Sampler, SamplerBindingType, ShaderStages,
    SpecializedMeshPipeline, SpecializedMeshPipelineError, SpecializedMeshPipelines,
    SpecializedRenderPipeline, TextureSampleType, TextureViewDescriptor, VertexAttribute,
    VertexBufferLayout, VertexFormat, VertexState, VertexStepMode,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::texture::{DefaultImageSampler, GpuImage, ImageSampler, TextureFormatPixelInfo};
use bevy::render::view::{ExtractedView, NoFrustumCulling, ViewUniform};
use bevy::render::{Extract, Render, RenderApp, RenderSet};
use std::borrow::Cow;

use crate::{map::*, tiles::*};

pub(crate) fn plugin(app: &mut App) {
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        warn!("Failed to get render app for tilemap_renderer");
        return;
    };

    render_app.add_systems(ExtractSchedule, extract_tilemap_chunks);
}

// ideally we don't want to have to copy the entire tilemap from the cpu to the render world.
// not when we can just copy over the changes, so this runs right where the render world copies
// stuff from the main world.

// It might make sense to have another step after this that runs entirely in the render
// world

fn extract_tilemap_chunks(
    // I *think that queries can be done line this... correct me if I am wrong here though.
    mut q_map_main: Query<(Entity, &TilemapChunks, &TilemapSize)>,
    mut q_map: Query<(&TilemapChunks, &TilemapSize)>,
    q_tile: Extract<
        Query<
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
    >,
    mut commands: Commands,
) {
    // first, init things if necessary
    for (entity, main_chunks, main_size) in q_map_main.iter() {
        if !main_chunks.is_added() {
            continue;
        }
        let mut tile_map_chunk_entity = commands.get_or_spawn(entity);
        tile_map_chunk_entity.insert((main_chunks, main_size))
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
    pipeline_id: CachedRenderPipelineId,
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
                (
                    texture_2d(TextureSampleType::Uint),
                    sampler(SamplerBindingType::Filtering),
                ),
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
            pipeline_id: (),
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
        RenderPipelineDescriptor {
            label: Some(Cow::from("TileMapPipeline")),
            layout: vec![],
            push_constant_ranges: vec![],
            vertex: VertexState {
                shader: self.vertex_shader.clone(),
                shader_defs: vec![],
                entry_point: Default::default(),
                buffers: vec![],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(FragmentState {
                shader: self.fragment_shader.clone(),
                shader_defs: vec![],
                entry_point: Default::default(),
                targets: vec![],
            }),
        }
    }
}

// Render everything.
// Copy pasted from bevies sprite renderer for reference
// todo: replace with TileMap specifics

// Also need a perpare system step possibly in the render world to format things
// for best efficiency.

// Alternatively we imple Node and fully customize the rendering step?
// though seems like we can make use of the PhaseItem and RenderCommand machinery
struct DrawTileMap {}
impl<P: PhaseItem> RenderCommand<P> for DrawTileMap {
    type Param = SRes<SpriteMeta>;
    type ViewQuery = ();
    type ItemQuery = Read<SpriteBatch>;

    fn render<'w>(
        _item: &P,
        _view: (),
        batch: Option<&'_ SpriteBatch>,
        sprite_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let sprite_meta = sprite_meta.into_inner();
        let Some(batch) = batch else {
            return RenderCommandResult::Failure;
        };

        pass.set_index_buffer(
            sprite_meta.sprite_index_buffer.buffer().unwrap().slice(..),
            0,
            IndexFormat::Uint32,
        );
        pass.set_vertex_buffer(
            0,
            sprite_meta
                .sprite_instance_buffer
                .buffer()
                .unwrap()
                .slice(..),
        );
        pass.draw_indexed(0..6, 0, batch.range.clone());
        RenderCommandResult::Success
    }
}
