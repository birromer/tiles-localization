-- u1 : moteur avant gauche Join_Left_Front frontMotorLeft
-- u2 : moteur avant droit Join_Right_Front frontMotorRight
-- u3 : moteur arriere gauche Join_Left_Back backMotorLeft
-- u4 : moteur arriere droit Join_Right_Back backMotorRight

-- function subscriber_cmd_u1_callback(msg)
--     spdMotor = msg["data"]
--     sim.setJointTargetVelocity(frontMotorLeft, spdMotor)
--   sim.addStatusbarMessage('cmd_u1 subscriber receiver : wheels speed ='..spdMotor)
-- end

-- function subscriber_cmd_u2_callback(msg)
--   spdMotor = msg["data"]
--   sim.setJointTargetVelocity(frontMotorRight, spdMotor)
--   sim.addStatusbarMessage('cmd_u1 subscriber receiver : wheels speed ='..spdMotor)
-- end

-- function subscriber_cmd_u3_callback(msg)
--   spdMotor = msg["data"]
--   sim.setJointTargetVelocity(backMotorLeft, spdMotor)
--   sim.addStatusbarMessage('cmd_u1 subscriber receiver : wheels speed ='..spdMotor)
-- end

-- function subscriber_cmd_u4_callback(msg)
--   spdMotor = msg["data"]
--   sim.setJointTargetVelocity(backMotorRight, spdMotor)
--   sim.addStatusbarMessage('cmd_u1 subscriber receiver : wheels speed ='..spdMotor)
-- end

function gaussian (mean, variance)
    return  math.sqrt(-2 * variance * math.log(math.random())) *
            math.cos(2 * math.pi * math.random()) + mean
end

function getCompass(objectName, statemotor)
  -- This function get the value of the compass
  objectHandle = sim.getObjectAssociatedWithScript(sim.handle_self)
  relTo = -1
  o = sim.getObjectOrientation(objectHandle, relTo)
  if statemotor then -- if motor is ON
    heading = o[3] + gaussian(0,1)*math.pi  -- gaussian noise is added
  else
    heading = o[3]   -- north along X > 0
  end
  return heading -- in radians
--  return heading*180/math.pi -- in degrees
end

function getSpeed(objectName)
  -- This function get the object pose at ROS format geometry_msgs/Pose
  objectHandle = sim.getObjectAssociatedWithScript(sim.handle_self)
  relTo = -1
  p, o =sim.getObjectVelocity(objectHandle)

  return {
    position={x=p[1],y=p[2],z=p[3]},
    orientation={x=o[1],y=o[2],z=o[3],w=0}
  }
end

function subscriber_cmd_ul_callback(msg)
  spdMotor = msg["data"]
  sim.setJointTargetVelocity(backMotorLeft, spdMotor)
  sim.setJointTargetVelocity(frontMotorLeft, spdMotor)

  sim.addStatusbarMessage('cmd_u1 subscriber receiver : wheels speed ='..spdMotor)
end

function subscriber_cmd_ur_callback(msg)
  spdMotor = msg["data"]
  sim.setJointTargetVelocity(backMotorRight, spdMotor)
  sim.setJointTargetVelocity(frontMotorRight, spdMotor)

  sim.addStatusbarMessage('cmd_u1 subscriber receiver : wheels speed ='..spdMotor)
end

function getPose(objectName)
  -- This function get the object pose at ROS format geometry_msgs/Pose
  objectHandle=sim.getObjectHandle(objectName)
  relTo = -1
  p=sim.getObjectPosition(objectHandle,relTo)
  o=sim.getObjectQuaternion(objectHandle,relTo)

  return {
    position={x=p[1],y=p[2],z=p[3]},
    orientation={x=o[1],y=o[2],z=o[3],w=o[4]}
  }
end

function getTransformStamped(objHandle,name,relTo,relToName)
  -- This function retrieves the stamped transform for a specific object
  t=sim.getSystemTime()
  p=sim.getObjectPosition(objHandle,relTo)
  o=sim.getObjectQuaternion(objHandle,relTo)

  return {
    header={
      stamp=t,
      frame_id=relToName
    },
    child_frame_id=name,
    transform={
      translation={x=p[1],y=p[2],z=p[3]},
      rotation={x=o[1],y=o[2],z=o[3],w=o[4]}
    }
  }
end

function sysCall_init()
  -- The child script initialization
  objectName = "Chassis"
  objectHandle = sim.getObjectHandle(objectName)

  -- get left and right motors handles
  backMotorLeft = sim.getObjectHandle("Join_Left_Back")
  backMotorRight = sim.getObjectHandle("Join_Right_Back")
  frontMotorLeft = sim.getObjectHandle("Join_Left_Front")
  frontMotorRight = sim.getObjectHandle("Join_Right_Front")

  rosInterfacePresent = simROS

  -- Prepare the publishers and subscribers :
  if rosInterfacePresent then
    publisher_time    = simROS.advertise('/simulationTime','std_msgs/Float64')
    publisher_pose    = simROS.advertise('/pose','geometry_msgs/Pose')
    publisher_speed   = simROS.advertise('/speed','geometry_msgs/Pose')
    publisher_compass = simROS.advertise('/compass','std_msgs/Float64')

    subscriber_ul = simROS.subscribe('/cmd_l','std_msgs/Float64','subscriber_cmd_ul_callback')
    subscriber_ur = simROS.subscribe('/cmd_r','std_msgs/Float64','subscriber_cmd_ur_callback')
  end

  -- Get some handles (as usual !):
  onboard_camera = sim.getObjectHandle('OnBoardCamera')

  -- Enable an image publisher and subscriber:
  pub = simROS.advertise('/image', 'sensor_msgs/Image')
  simROS.publisherTreatUInt8ArrayAsString(pub) -- treat uint8 arrays as strings (much faster, tables/arrays are kind of slow in Lua)
end

function sysCall_sensing()
    -- Publish the image of the vision sensor:
    local data, w, h = sim.getVisionSensorCharImage(onboard_camera)
    d = {}
    d['header'] = {seq=0,stamp=simROS.getTime(), frame_id="a"}
    d['height'] = h
    d['width'] = w
    d['encoding'] = 'rgb8'
    d['is_bigendian'] = 1
    d['step'] = w*3
    d['data'] = data
    simROS.publish(pub,d)

    simROS.publish(publisher_compass, {data=getCompass(sim.handle_self, statemotor)})
end

function sysCall_actuation()
   -- Send an updated simulation time message, and send the transform of the object attached to this script:
   if rosInterfacePresent then
      -- publish time and pose topics
      simROS.publish(publisher_time, {data=sim.getSimulationTime()})
      simROS.publish(publisher_pose, getPose("Chassis"))
      simROS.publish(publisher_speed, getSpeed(sim.handle_self))

      -- send a TF
      -- To send several transforms at once, use simROS.sendTransforms instead
   end
end

function sysCall_cleanup()
    -- Following not really needed in a simulation script (i.e. automatically shut down at simulation end):
  if rosInterfacePresent then
    simROS.shutdownPublisher(publisher_time)
    simROS.shutdownPublisher(publisher_pose)
    simROS.shutdownPublisher(publisher_speed)
    simROS.shutdownPublisher(publisher_compass)

    simROS.shutdownSubscriber(subscriber_ul)
    simROS.shutdownSubscriber(subscriber_ur)

    simROS.shutdownPublisher(pub)
  end
end
