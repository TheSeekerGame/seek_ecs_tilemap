use bevy::prelude::*;

use crate::{map::*, tiles::*};

pub(crate) fn plugin(app: &mut App) {
}

fn update_tilemap_chunks(
    mut q_map: Query<(
        &mut TilemapChunks,
        &TilemapSize,
    )>,
    q_tile: Query<(
        &TilemapId,
        &TilePos,
        &TileTextureIndex,
        &TileColor,
        &TileFlip,
        &TileVisible,
    ), Or<(
        Changed<TileTextureIndex>,
        Changed<TileColor>,
        Changed<TileFlip>,
        Changed<TileVisible>,
    )>>,
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
