use bevy::prelude::*;
use seek_ecs_tilemap::map::{
    TilemapChunks, TilemapGridSize, TilemapSize, TilemapTileSize, TilemapType,
};
use seek_ecs_tilemap::{TilemapBundle, TilemapPlugin};

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins);
    app.add_plugins(TilemapPlugin);
    app.add_systems(Startup, setup);
    app.run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());
    // todo finish spawning a TilemapTest that functions
    commands.spawn(TilemapBundle {
        grid_size: TilemapGridSize::new(20480.0, 20480.0),
        map_type: TilemapType::Square,
        size: TilemapSize::new(2048, 2048),
        spacing: Default::default(),
        storage: Default::default(),
        tile_size: TilemapTileSize::new(10.0, 10.0),
        chunks: TilemapChunks::default(),
        transform: Default::default(),
        global_transform: Default::default(),
        visibility: Default::default(),
        inherited_visibility: Default::default(),
        view_visibility: Default::default(),
    });
}
