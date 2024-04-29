import asyncio
from mavsdk import System
from mavsdk.offboard import (OffboardError, VelocityNedYaw)

async def initialize_drone(drone: System):
    try:
        await drone.connect(system_address="serial:///dev/ttyUSB0:921600")

        print("Waiting for drone to connect...")
        async for state in drone.core.connection_state():
            if state.is_connected:
                print(f"Drone discovered!")
                break
        async for health in drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print("Global position and home position ok")
                break

        print("-- Arming")
        await drone.action.arm()
        await asyncio.sleep(0.01)

        print("-- Taking off")
        await drone.action.set_takeoff_altitude(2)
        await drone.action.takeoff()
        await asyncio.sleep(1)

        print("-- Starting offboard")
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))  # Send an initial setpoint before starting offboard mode
        await drone.offboard.start()
        print("-- Offboard started")
        print("-- Setting initial setpoint")
    except Exception as e:
        print(f"An error occurred: {e}")
        try:
            await drone.offboard.stop()
        except OffboardError as error:
            print(f"Stopping offboard mode failed with error code: {error._result.result}")
        await drone.action.return_to_launch()

async def main():
    drone = System()
    try:
        await initialize_drone(drone)
        await asyncio.sleep(1)
    finally:
        print("-- Landing")
        await drone.action.land()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
