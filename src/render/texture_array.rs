use bevy::prelude::default;
use bevy::render::render_resource::{AddressMode, Extent3d, FilterMode, Sampler, SamplerDescriptor, Texture, TextureDescriptor, TextureDimension, TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use crate::render::ExtractedTileset;

struct TextureArray {
    texture: Texture,
    pub(crate) view: TextureView,
    pub(crate) sampler: Sampler,
}

pub fn create_texture_array(
    device: &RenderDevice,
    queue: &RenderQueue,
    extracted_texture: &ExtractedTileset,
) -> TextureArray {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("tilemap_texture_array"),
        size: Extent3d {
            width: extracted_texture.texture_size.x as u32,
            height: extracted_texture.texture_size.y as u32,
            depth_or_array_layers: extracted_texture.tile_count,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: extracted_texture.format,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let view = texture.create_view(&TextureViewDescriptor {
        label: Some("tilemap_texture_array_view"),
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });

    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("tilemap_texture_array_sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: extracted_texture.filtering,
        min_filter: extracted_texture.filtering,
        mipmap_filter: FilterMode::Nearest,
        ..default()
    });

    // todo:
    // queue.write_texture()

    TextureArray {
        texture,
        view,
        sampler,
    }
}