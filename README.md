# isaacgymbench
a new benchmark of mujoco tasks of gym repo using isaacgym platform

# bugs

hopper env observation bugs check each component later 

walker env meets some bugs under analysis

```
# for j in range(self.num_bodies):
#     self.gym.set_rigid_body_color(
#         env_ptr, walker_asset, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))


    1. (self: isaacgym._bindings.linux-x86_64.gym_37.Gym, arg0: isaacgym._bindings.linux-x86_64.gym_37.Env, arg1: int, arg2: int, arg3: isaacgym._bindings.linux-x86_64.gym_37.MeshType, arg4: isaacgym._bindings.linux-x86_64.gym_37.Vec3) -> None

    Invoked with: <isaacgym._bindings.linux-x86_64.gym_37.Gym object at 0x7f3741b5fa30>, <isaacgym._bindings.linux-x86_64.gym_37.Env object at 0x7f3741b4d4b0>, <isaacgym._bindings.linux-x86_64.gym_37.Asset object at 0x7f37430e3670>, 0, MeshType.MESH_VISUAL, <isaacgym._bindings.linux-x86_64.gym_37.Vec3 object at 0x7f3741b4d6b0>

```
