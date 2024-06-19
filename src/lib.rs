use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::sprite::SpritePipeline;

/// A module which contains tilemap components.
pub mod map;
#[cfg(feature = "render")]
pub(crate) mod render;
/// A module which contains tile components.
pub mod tiles;

pub use crate::map::TilemapBundle;

pub struct TilemapPlugin;

impl Plugin for TilemapPlugin {
    fn build(&self, app: &mut App) {
        use crate::map::*;
        use crate::tiles::*;
        app.register_type::<TilemapId>()
            .register_type::<TilemapSize>()
            // .register_type::<TilemapTexture>()
            .register_type::<TilemapTileSize>()
            .register_type::<TilemapGridSize>()
            .register_type::<TilemapSpacing>()
            .register_type::<TilemapTextureSize>()
            .register_type::<TilemapType>()
            .register_type::<TilePos>()
            .register_type::<TileTextureIndex>()
            .register_type::<TileColor>()
            .register_type::<TileVisible>()
            .register_type::<TileFlip>()
            .register_type::<TileStorage>()
            .register_type::<TilePosOld>();

        app.add_plugins((crate::map::plugin, crate::tiles::plugin));
        #[cfg(feature = "render")]
        app.add_plugins((
            //crate::chunk::plugin,
            crate::render::TileMapRendererPlugin,
        ));
    }
}
