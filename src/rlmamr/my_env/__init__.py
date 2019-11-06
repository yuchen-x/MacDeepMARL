#!/usr/bin/python

from gym.envs.registration import register

"""    1) no macro_action, each action can be done in a single step,
       2) distiguish the Look_For_Obj and Pass_Obj action to different Turtlebots
       3) fetch doesn't remember communication message about which object turtlebot need
       4) two prefined locations for each turtlebot beside the table
       5) fetch won't pass turtlebot_1's obj to turtlebot_0"""

register(
    id='ObjSearchDelivery-v0',
    entry_point='my_env.ObjSearchDelivery_v0:ObjSearchDelivery',
    )

"""    1) no macro_action, each action can be done in a single step,
       2) distiguish the Look_For_Obj and Pass_Obj action to different Turtlebots
       3) fetch doesn't remember communication message about which object turtlebot need
       4) two prefined locations for each turtlebot beside the table
       5) fetch won't pass turtlebot_1's obj to turtlebot_0
       6) IMPORTANT: turtlebots observe human's status instead of request obj"""
       

register(
    id='ObjSearchDelivery-v1',
    entry_point='my_env.ObjSearchDelivery_v1:ObjSearchDelivery',
    )

"""    1) macro_action
       2) distiguish the Look_For_Obj and Pass_Obj action to different Turtlebots
       3) fetch doesn't remember communication message about which object turtlebot need
       4) two prefined locations for each turtlebot beside the table
       5) fetch won't pass turtlebot_1's obj to turtlebot_0

       Q: does fetch have to observe which turtlebot in the room?"""
 

register(
    id='ObjSearchDelivery-MA-v0',
    entry_point='my_env.ObjSearchDelivery_MA_v0:ObjSearchDelivery',
    )

"""    1) macro_action
       2) distiguish the Look_For_Obj and Pass_Obj action to different Turtlebots
       3) fetch doesn't remember communication message about which object turtlebot need
       4) two prefined locations for each turtlebot beside the table
       5) fetch won't pass turtlebot_1's obj to turtlebot_0
       6) IMPORTANT: turtlebots observe human's status instead of request obj

       Q: does fetch have to observe which turtlebot in the room?"""
 

register(
    id='ObjSearchDelivery-MA-v1',
    entry_point='my_env.ObjSearchDelivery_MA_v1:ObjSearchDelivery',
    )

register(
    id='BoxPushing-MA-v0',
    entry_point='my_env.box_pushing_MA:BoxPushing',
    )







