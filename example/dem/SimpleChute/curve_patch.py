from geotaichi import *

init(arch="gpu", debug=False)


dem = DEM()

dem.set_configuration(domain=ti.Vector([0.6, 0.4, 0.55]), 
                          boundary=["Destroy", "Destroy", "Destroy"],
                          gravity=ti.Vector([0., 0., -9.8]),  
                          engine="VelocityVerlet",
                          search="LinkedCell")

dem.memory_allocate(memory={
        "max_material_number": 1,
        "max_particle_number": 9000,  
        "max_sphere_number": 9000,   
        "max_clump_number": 0,
        "max_patch_number": 1616,
        "max_servo_wall_number": 0,
        "body_coordination_number": 32,
        "wall_coordination_number": 64,
        "verlet_distance_multiplier": 0.4,
        "compaction_ratio": [1., 1.]
    })

dem.set_solver({
        "Timestep": 1e-5,
        "SimulationTime": 1.01,
        "SaveInterval": 0.04
    })


dem.add_attribute(materialID=0,
                      attribute={
                          "Density": 2650, 
                          "ForceLocalDamping": 0.05,
                          "TorqueLocalDamping": 0.
                      })
                      
dem.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.465, 0.165, 0.54]),
                            "BoundingBoxSize": ti.Vector([0.07, 0.07, 0.01]),
                            "zdirection": ti.Vector([0., 0., 1.])
                      }])

dem.add_body(body={
                    "BodyType": "Sphere",
                    "GenerateType": "Generate",
                    "Period": [0, 1, 0.05], 
                    "RegionName": "region1",
                    "Template":[{
                                "GroupID": 0,
                                "MaterialID": 0,
                                "InitialVelocity": ti.Vector([0., 0., 0.]),
                                "InitialAngularVelocity": ti.Vector([0., 0., 0.]),
                                "FixVelocity": ["Free","Free","Free"],
                                "FixAngularVelocity": ["Free","Free","Free"],
                                "Radius": 0.0015,
                                "BodyNumber": 450,
                                "BodyOrientation": "uniform"}]})
                                
dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   0,
        "MaterialID":   0,
        "Counterclockwise":  False,
        "WallFile":'simple_chute.stl',
        "Translation": [0.5, 0.2, 0.4],
        "Visualize": True
    })
    
dem.choose_contact_model(particle_particle_contact_model="Hertz Mindlin Model",
                             particle_wall_contact_model="Hertz Mindlin Model")

dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "ShearModulus":               5e6,
                            "Possion":                    0.45,
                            "Friction":                   0.5,
                            "Restitution":                0.3,   
                           }
                 )

dem.select_save_data(sphere=True, wall=True, particle_particle_contact=True, particle_wall_contact=True)

dem.run()
