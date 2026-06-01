#!/usr/bin/env python3
import time
import logging
import numpy as np

from core.config import HAND_TYPE, HAND_JOINT, CAN_INTERFACE
from control.linkerhand_python_sdk.LinkerHand.linker_hand_api import LinkerHandApi

logger = logging.getLogger(__name__)

class HandMotionPlanner:
    def __init__(self, verbose=True):
        self.verbose = verbose
        
        if self.verbose:
            logger.info("Initializing HandMotionPlanner")
        #Constants
        self.GRASP_POSES = [
            "PalmarPinch",      #Thumb-index pinch
            "2FingerPinch",     #Thumb-2 finger
            "LateralPinch",     #Key grip
            
            #Power
            "Wrap",             #Dynamic wrap
        ]

        self.POSES = {
            #Precision
            "PalmarPinch": [70, 0, 140, 0, 0, 0, 210],
            "2FingerPinch": [80, 0, 140, 130, 0, 0, 190],
            "LateralPinch": [0, 255, 30, 0, 0, 0, 255],
            
            #Power
            "Wrap": [10, 0, 0, 0, 0, 0, 170],
            "Hook": [120, 255, 120, 110, 110, 110, 200],
            "Open": [255, 255, 255, 255, 255, 255, 255],
            
            #Default
            "Relaxed": [40, 110, 150, 140, 140, 140, 210],
            
        }

        self.KEYS = list(self.POSES.keys())

        self.MOTION_MATRIX = [
            #0  1  2  3  4  5  6
            [0, 2, 3, 3, 0, 3, 0],  #0 Pinch index
            [0, 0, 3, 3, 0, 0, 0],  #1 Pinch 2 finger
            [0, 0, 0, 3, 3, 0, 0],  #2 Lateral pinch
            [3, 3, 3, 0, 3, 0, 3],  #3 Wrap 
            [0, 0, 2, 2, 0, 0, 0],  #4 Hook
            [0, 0, 2, 3, 0, 0, 0],  #5 Open
            [0, 0, 2, 2, 0, 0, 0],  #6 Relaxed
        ]
        
        self.THUMB_JOINTS = [0, 1, 6]
        self.FINGER_JOINTS = [2, 3, 4, 5]
        
        #Hand API
        hand = LinkerHandApi(
            hand_joint=HAND_JOINT,
            hand_type=HAND_TYPE,
            can=CAN_INTERFACE
            )
        self.hand = hand
        
        #Initialize hand
        self.current_pose = "Open"

        self.hand.set_speed([255]*7)
        self.hand.set_torque([255]*7)

        self.hand.finger_move(self.POSES[self.current_pose])
        
        if self.verbose:
            logger.info(f"HandMotionPlanner ready, current pose: {self.current_pose}")

    # -----------------------------
    # Action selection
    # -----------------------------
    def action(self,command):
        if command in self.GRASP_POSES:
            self.execute_grasp(command)
        else:
            self.move(command)

    # -----------------------------
    # BASIC MOTION
    # -----------------------------
    def move(self, to_gesture):
        if self.current_pose ==  to_gesture:
            if self.verbose:
                logger.info("No move - already at target pose")
            return
    
        if self.current_pose in self.GRASP_POSES and to_gesture != "Open":
            if self.verbose:
                logger.info("Must open before new grasp")
            return
        else:
            if self.verbose:
                logger.info("Releasing grasp")
            self.release()
        
        target_pose = self.POSES[to_gesture]
        rule = self.MOTION_MATRIX[self.KEYS.index(self.current_pose)][self.KEYS.index(to_gesture)]

        print(f"\nPlanning move: {self.current_pose} → {to_gesture} (rule {rule})")
        print(f"Moving to: {target_pose}\n")
        if rule == 0:
            self._direct(target_pose)
        elif rule == 1:
            self._thumb_first(target_pose)
        elif rule == 2:
            self._thumb_last(target_pose)
        elif rule == 3:
            self._thumb_away(target_pose)

        self.current_pose = to_gesture

    def _wait_until_reached(self, target, tol=5, timeout=2.0, poll_dt=0.02):
        start = time.time()
        target = np.array(target)

        while True:
            # print(f"DEBUG state: {self.hand.get_state()}")
            state = np.array(self.hand.get_state())
            error = np.abs(state - target)

            if np.max(error) <= tol:
                if self.verbose:
                    logger.info("Motion complete")
                return  # Reached position

            if time.time() - start > timeout:
                if self.verbose:
                    logger.info("Motion timeout")
                return

            time.sleep(poll_dt)

    def _direct(self, target):
        self.hand.finger_move(target)
        self._wait_until_reached(target)

    def _thumb_first(self, target):
        # Step 1: Move thumb only
        intermediate = self.POSES[self.current_pose].copy()

        #Move thumb
        for j in self.THUMB_JOINTS:
            intermediate[j] = target[j]

        # Release fingers
        for j in self.FINGER_JOINTS:
            intermediate[j] = min(intermediate[j]+20,255)

        print("Step 1 → Moving thumb first")
        self.hand.finger_move(intermediate)
        self._wait_until_reached(intermediate)

        # Step 2: Move full target
        print("Step 2 → Moving remaining fingers")
        self.hand.finger_move(target)
        self._wait_until_reached(target)

    def _thumb_last(self, target):
        # Step 1: Move fingers only
        intermediate = self.POSES[self.current_pose].copy()

        #Move fingers
        for j in self.FINGER_JOINTS:
            intermediate[j] = target[j]

        # Release thumb
        for j in self.THUMB_JOINTS:
            intermediate[j] = min(intermediate[j]+20,255)

        print("Step 1 → Moving fingers first")
        self.hand.finger_move(intermediate)
        self._wait_until_reached(intermediate)

        # Step 2: Move full target
        print("Step 2 → Moving thumb")
        self.hand.finger_move(target)
        self._wait_until_reached(target)

    def _thumb_away(self, target):
        # Step 1: Move thumb out of way
        intermediate = self.POSES[self.current_pose].copy()

        #Move thumb
        intermediate[0] = 255

        # Release fingers
        for j in self.FINGER_JOINTS:
            intermediate[j] = min(intermediate[j]+20,255)

        print("Step 1 → Moving thumb out of way")
        self.hand.finger_move(intermediate)
        self._wait_until_reached(intermediate)

        #Move fingers
        for j in self.FINGER_JOINTS:
            intermediate[j] = target[j]

        # Step 2: Move fingers
        print("Step 2 → Moving fingers")
        self.hand.finger_move(intermediate)
        self._wait_until_reached(intermediate)

        # Step 3: Rotate thumb
        intermediate[1] = target[1]

        print("Step 3 → Rotate thumb")
        self.hand.finger_move(intermediate)
        self._wait_until_reached(intermediate)

        # Step 4: Move full target
        print("Step 4 → Moving thumb back")
        self.hand.finger_move(target)
        self._wait_until_reached(target)

    # -----------------------------
    # GRASP EXECUTION
    # -----------------------------
    def execute_grasp(self, gesture):
        if self.current_pose in self.GRASP_POSES and gesture in self.GRASP_POSES:
            if self.verbose:
                logger.info("Must open before new grasp")
            return
        print(f"\nExecuting grasp: {gesture}")

        target = self.POSES[gesture]

        print("→ Pre-grasp")
        self._pre_move(gesture)
        time.sleep(0.75)

        print("→ Closing until contact")
        self._close_until_contact(target)

        print("→ Hold (torque limited)")
        self._hold_force(duration=2.0)

        self.current_pose = gesture

    def _pre_move(self, to_gesture):
        if self.current_pose ==  to_gesture:
            print(f"\nMove to same position → no move")
            return
        
        target_pose = self._make_pregrasp(self.POSES[to_gesture],to_gesture)
        # print(f"DEBUG pre grasp pose {target_pose}")
        rule = self.MOTION_MATRIX[self.KEYS.index(self.current_pose)][self.KEYS.index(to_gesture)]

        print(f"\nPlanning pre-move: {self.current_pose} → {to_gesture} (rule {rule})")
        print(f"Moving to: {target_pose}\n")
        if rule == 0:
            self._direct(target_pose)
        elif rule == 1:
            self._thumb_first(target_pose)
        elif rule == 2:
            self._thumb_last(target_pose)
        elif rule == 3:
            self._thumb_away(target_pose)

        self.current_pose = to_gesture

    def _make_pregrasp(self, pose, gesture):
        pre = pose.copy()
        open_delta = [60,0,30,30,30,30,0]
        if gesture == "Wrap":
            open_delta[0] = 200
        if gesture == "Lateral pinch":
            open_delta[6] = 60
        if gesture == "Pinch 2 finger":
            open_delta[2] *= 2
            open_delta[3] *= 2

        for i in range(len(pre)):
            pre[i] = min(pre[i] + open_delta[i], 255)
        return pre

    def _close_until_contact(self,
                            target,
                            step=5,
                            poll_dt=0,
                            stall_eps=2,
                            stall_count_max=3):

        target = np.array(target)
        current = np.array(self.hand.get_state())
        stall_count = 0
        contact_seen = False
        
        if self.verbose:
            logger.info(f"Closing until contact - target: {target}, step: {step}")

        while True:
            touch = np.array(self.hand.get_touch())
            # print(f"DEBUG touch: {touch}")
            if np.any(touch > 50):
                contact_seen = True

            # Move *towards target*, not blindly close
            next_cmd = current - step

            # Clamp so we never pass the target
            for j in range(len(next_cmd)):
                next_cmd[j] = max(next_cmd[j], target[j])

            self.hand.finger_move(next_cmd.tolist())
            time.sleep(poll_dt)

            new_state = np.array(self.hand.get_state())
            motion = np.abs(new_state - current)

            if np.max(motion) < stall_eps:
                stall_count += 1
            else:
                stall_count = 0

            # Stop when object is encountered AND motion is exhausted
            if contact_seen or stall_count >= stall_count_max:
                #Stops motors trying to run
                self.hand.finger_move(self.hand.get_state())
                if self.verbose:
                    logger.info("Grasp complete (blocked before target)")
                return

            # Or if target reached cleanly
            if np.max(np.abs(new_state - target)) < stall_eps:
                if self.verbose:
                    logger.info("Grasp complete (target reached)")
                return

            current = new_state

    def _hold_force(self, duration=0.1):
        self.hand.set_torque([120]*7)
        time.sleep(duration)

    def release(self):
        print("→ Release")
        self.hand.set_torque([255]*7)

if __name__ == "__main__":
    hand = LinkerHandApi(
        hand_joint=HAND_JOINT,
        hand_type=HAND_TYPE,
        can=CAN_INTERFACE
    )

    planner = HandMotionPlanner(hand)
    print("Hand ready")

    while True:
        #Get command - check if valid
        while True:
            cmd = input(f"Enter pose {planner.KEYS} or 'exit': ").strip()
            if cmd in planner.POSES or cmd == "exit":
                print()
                break

        if cmd == "exit":
            print("Exiting program")
            break

        elif cmd in planner.GRASP_POSES:
            planner.execute_grasp(cmd)

        elif cmd in planner.POSES:
            planner.move(cmd)