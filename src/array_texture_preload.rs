use std::sync::{Arc, RwLock};
use bevy::prelude::{Assets, Component, Handle, Image, Reflect, Res, ResMut, Resource};
use bevy::render::Extract;
use bevy::render::render_resource::{FilterMode, TextureFormat};
use bevy::render::texture::BevyDefault;
use crate::map::{TilemapSpacing, TilemapTileSize};

/*
#[derive(Component, Clone, Debug, Hash, PartialEq, Eq)]
pub struct TilemapTexture(Handle<Image>);

#[derive(Debug, Clone)]
pub struct TilemapArrayTexture {
    pub texture: TilemapTexture,
    pub tile_size: TilemapTileSize,
    pub tile_spacing: TilemapSpacing,
    /// Defaults to ImageSettings.
    pub filter: Option<FilterMode>,
    pub format: TextureFormat,
}

impl Default for TilemapArrayTexture {
    fn default() -> Self {
        Self {
            texture: Default::default(),
            tile_size: Default::default(),
            tile_spacing: Default::default(),
            filter: Default::default(),
            format: BevyDefault::bevy_default(),
        }
    }
}
/// A bevy world resource that allows you to add atlas textures for
/// loading into the array texture cache.
#[derive(Resource, Default, Debug, Clone)]
pub struct ArrayTextureLoader {
    // Arc and RwLock just let us get around Extract's read only rules.
    textures: Arc<RwLock<Vec<TilemapArrayTexture>>>,
}

impl ArrayTextureLoader {
    pub fn add(&self, texture: TilemapArrayTexture) {
        if let Ok(mut textures) = self.textures.try_write() {
            textures.push(texture);
        }
    }

    pub(crate) fn drain(&self) -> Vec<TilemapArrayTexture> {
        if let Ok(mut textures) = self.textures.try_write() {
            return std::mem::take(&mut *textures);
        }
        vec![]
    }
}


pub(crate) fn extract(
    images: Extract<Res<Assets<Image>>>,
    array_texture_loader: Extract<Res<ArrayTextureLoader>>,
    mut texture_array_cache: ResMut<TextureArrayCache>,
    default_image_settings: Res<DefaultSampler>,
) {
    for mut array_texture in array_texture_loader.drain() {
        if array_texture.filter.is_none() {
            array_texture
                .filter
                .replace(default_image_settings.mag_filter.into());
        }
        if array_texture.texture.verify_ready(&images) {
            texture_array_cache.add_texture(
                array_texture.texture,
                array_texture.tile_size,
                array_texture.tile_spacing,
                default_image_settings.min_filter.into(),
                array_texture.format,
                &images,
            );
        } else {
            // Image hasn't loaded yet punt to next frame.
            array_texture_loader.add(array_texture);
        }
    }
}*/
