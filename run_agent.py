import sys
import argparse
import numpy as np

from doom.actions import ActionBuilder
from doom.game import Game
from doom.utils import register_game_args
from src.utils import bool_flag, register_model_args
from src.agent import Agent

import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(description='LUBAN runner')
    register_model_args(parser)
    register_game_args(parser)
    params, unparsed = parser.parse_known_args(sys.argv)
    params.game_variables = [('health', 101), ('sel_ammo', 301)]

    params.frame_skip = 3
    params.eval_time = 900
    
    sess = tf.Session()
    agent = Agent(sess, params)
    agent.load("./checkpoint")

    action_builder = ActionBuilder(params)
    game = Game(
        scenario='full_deathmatch',
        action_builder=action_builder,
        score_variable='USER2',
        freedoom=True,
        screen_resolution='RES_400X225',
        screen_format='CRCGCB',
        use_screen_buffer=True,
        use_depth_buffer=False,
        labels_mapping="",
        game_features="target,enemy",
        mode=('SPECTATOR' if params.human_player else 'PLAYER'),
        render_hud=False,
        render_crosshair=True,
        render_weapon=True,
        freelook=params.freelook,
        visible=0,
        n_bots=4,
        use_scripted_marines=True
    )
    game.start(map_id=3)
    game.init_bots_health(100)
    
    n_iter = 0
    last_states = []
    # eval_time: seconds, 35: fps, eval_time*35: total frames
    while n_iter * params.frame_skip < params.eval_time * 35:
        n_iter += 1

        if game.is_player_dead():
            game.respawn_player()
            # TODO: add reset method to network
            # network.reset()
        
         # observe the game state / select the next action 
        game.observe_state(params, last_states)
        # single frame
        screen1 = last_states[11].screen.reshape((1, 1, 60, 108, 3))
        screen1 = screen1.astype(np.float32) / 255.
        # two frame
        screen2 = np.array([last_states[i+10].screen.transpose((1, 2, 0)) for i in range(2)]).reshape((1, 2, 60, 108, 3))
        screen2 = screen2.astype(np.float32) / 255.
        # four frame
        screen4 = np.array([last_states[i+8].screen.transpose((1, 2, 0)) for i in range(4)]).reshape((1, 4, 60, 108, 3))
        screen4 = screen4.astype(np.float32) / 255.
        # eight frame
        screen8 = np.array([last_states[i+4].screen.transpose((1, 2, 0)) for i in range(8)]).reshape((1, 8, 60, 108, 3))
        screen8 = screen8.astype(np.float32) / 255.
        # eight frame
        screen12 = np.array([last_states[i].screen.transpose((1, 2, 0)) for i in range(12)]).reshape((1, 12, 60, 108, 3))
        screen12 = screen12.astype(np.float32) / 255.
        action, output_gf = agent.choose_action(screen2)
        '''
        if action == 0:
            action = 5
        elif action == 1:
            action = 7
        elif action == 2:
            action = 9
        elif action == 3:
            action = 2
        elif action == 4:
            action = 4
        else:
            action = 0	
        '''
        print("target: %.4f, enemy: %.4f" % (output_gf[0][0], output_gf[0][1]))        
        #action = np.random.randint(0, 29)
        print("health: %d\tsel_ammo: %d\taction: %d" % (last_states[3].variables[0], last_states[3].variables[1], action))

        sleep = 0.01
        game.make_action(action, frame_skip=params.frame_skip, sleep=sleep)

    game.close()
    game.print_statistics()

if __name__ == '__main__':
    main()
