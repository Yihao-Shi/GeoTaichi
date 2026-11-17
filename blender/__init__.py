"""GeoTaichi Blender add-on: SolverJobs and animated result inspection."""

bl_info = {
    "name": "GeoTaichi SolverJob",
    "author": "GeoTaichi contributors",
    "version": (0, 3, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > GeoTaichi",
    "description": "Run GeoTaichi SolverJobs and inspect animated VTU result sequences",
    "category": "Physics",
}


def register():
    from . import operators, panels, properties, results

    properties.register()
    operators.register()
    results.register()
    panels.register()


def unregister():
    from . import operators, panels, properties, results

    panels.unregister()
    results.unregister()
    operators.unregister()
    properties.unregister()


if __name__ == "__main__":
    register()
