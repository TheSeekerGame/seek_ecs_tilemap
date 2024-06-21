use bevy::prelude::*;
use seek_ecs_tilemap::map::*;
use seek_ecs_tilemap::tiles::*;
use seek_ecs_tilemap::{TilemapBundle, TilemapPlugin};

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins);
    app.add_plugins(TilemapPlugin);
    app.add_systems(Startup, setup);
    app.run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2dBundle {
        transform: Transform::from_xyz(420.0, 240.0, 0.0),
        ..Default::default()
    });
    let width = 1503;
    let height = 1503;
    let e_tilemap = commands.spawn(TilemapBundle {
        grid_size: TilemapGridSize::new(10.0, 10.0),
        map_type: TilemapType::Square,
        size: TilemapSize::new(width, height),
        spacing: Default::default(),
        storage: Default::default(),
        tile_size: TilemapTileSize::new(10.0, 10.0),
        chunks: TilemapChunks::default(),
        transform: Default::default(),
        global_transform: Default::default(),
        visibility: Default::default(),
        inherited_visibility: Default::default(),
        view_visibility: Default::default(),
    }).id();
    for y in 0..height {
        for x in 0..width {
            commands.spawn(TileBundle {
                position: TilePos::new(x, y),
                texture_index: TileTextureIndex(0),
                tilemap_id: TilemapId(e_tilemap),
                visible: TileVisible(true),
                flip: TileFlip { x: false, y: false, d: false},
                color: TileColor(Srgba::rgb_u8((x % 256) as u8, (y % 256) as u8, 127).into()),
                old_position: default(),
            });
        }
    }
}
