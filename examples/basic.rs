use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::window::WindowMode;
use seek_ecs_tilemap::map::*;
use seek_ecs_tilemap::tiles::*;
use seek_ecs_tilemap::{TilemapBundle, TilemapPlugin};

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            present_mode: bevy::window::PresentMode::Immediate, // Disable vsync
            mode: WindowMode::Windowed,
            ..default()
        }),
        ..default()
    }));
    app.add_plugins(TilemapPlugin);
    app.add_systems(Startup, setup);
    app.add_plugins(LogDiagnosticsPlugin::default())
        .add_plugins(FrameTimeDiagnosticsPlugin::default());
    app.run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2dBundle {
        transform: Transform::from_xyz(420.0, 240.0, 0.0),
        ..Default::default()
    });
    let width = 100;
    let height = 10;
    let e_tilemap = commands
        .spawn(TilemapBundle {
            grid_size: TilemapGridSize::new(10.0, 10.0),
            map_type: TilemapType::Square,
            size: TilemapSize::new(width, height),
            spacing: Default::default(),
            storage: Default::default(),
            tile_size: TilemapTileSize::new(8.0, 8.0),
            chunks: TilemapChunks::default(),
            transform: Default::default(),
            global_transform: Default::default(),
            visibility: Default::default(),
            inherited_visibility: Default::default(),
            view_visibility: Default::default(),
        })
        .id();
    for y in 0..height {
        for x in 0..width {
            commands.spawn(TileBundle {
                position: TilePos::new(x, y),
                texture_index: TileTextureIndex(0),
                tilemap_id: TilemapId(e_tilemap),
                visible: TileVisible(y % 2 == 0 || x % 2 == 0),
                flip: TileFlip { x: false, y: false, d: false},
                color: TileColor(Srgba::rgb_u8(((x *25) % 256) as u8, ((y * 25) % 256) as u8, 0).into()),
                old_position: default(),
            });
        }
    }
}
