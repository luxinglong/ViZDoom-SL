import sys
import argparse
import numpy as np

from actions import ActionBuilder
from game import Game

# use_continuous speed action_combinations crouch freelook

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

def bool_flag(string):
    """
    Parse boolean arguments from the command line.
    """
    if string.lower() in FALSY_STRINGS:
        return False
    elif string.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag. "
                                         "use 0 or 1")

def main():
    parser = argparse.ArgumentParser(description='LUBAN runner')
    parser.add_argument("--use_continuous", type=bool_flag, default=False,
                        help="weather use continuous actions")
    # Available actions
    # combination of actions the agent is allowed to do.
    # this is for non-continuous mode only, and is ignored in continuous mode
    parser.add_argument("--action_combinations", type=str,
                        default='move_fb+turn_lr+move_lr+attack',
                        help="Allowed combinations of actions")
    # freelook: allow the agent to look up and down
    parser.add_argument("--freelook", type=bool_flag, default=False,
                        help="Enable freelook (look up / look down)")
    parser.add_argument("--human_player", type=bool_flag, default=False,
                        help="DoomGame mode")

    # speed and crouch buttons: in non-continuous mode, the network can not
    # have control on these buttons, and they must be set to always 'on' or
    # 'off'. In continuous mode, the network can manually control crouch and
    # speed.
    parser.add_argument("--speed", type=str, default='off',
                        help="Crouch: on / off / manual")
    parser.add_argument("--crouch", type=str, default='off',
                        help="Crouch: on / off / manual")

    # for process_buffers
    parser.add_argument("--height", type=int, default=60,
                        help="Image height")
    parser.add_argument("--width", type=int, default=108,
                        help="Image width")
    parser.add_argument("--gray", type=bool_flag, default=False,
                        help="Use grayscale")
    parser.add_argument("--use_screen_buffer", type=bool_flag, default=True,
                        help="Use the screen buffer")
    parser.add_argument("--use_depth_buffer", type=bool_flag, default=False,
                        help="Use the depth buffer")
    parser.add_argument("--labels_mapping", type=str, default='',
                        help="Map labels to different feature maps")
    parser.add_argument("--dump_freq", type=int, default=0,
                        help="Dump every X iterations (0 to disable)")
    # for observe_state
    parser.add_argument("--hist_size", type=int, default=4,
                        help="History size")

    params, unparsed = parser.parse_known_args(sys.argv)
    print(sys.argv)
    params.game_variables = [('health', 101), ('sel_ammo', 301)]
    print(params)

    action_builder = ActionBuilder(params)
    print(action_builder.n_actions)
    print(action_builder.available_actions)

    game = Game(
        scenario='full_deathmatch',
        action_builder=action_builder,
        score_variable='USER2',
        freedoom=True,
        screen_resolution='RES_800X450',
        use_screen_buffer=True,
        use_depth_buffer=True,
        labels_mapping="",
        game_features="target,enemy",
        mode=('SPECTATOR' if params.human_player else 'PLAYER'),
        render_hud=True,
        render_crosshair=True,
        render_weapon=True,
        freelook=params.freelook,
        visible=0,
        n_bots=10,
        use_scripted_marines=True
    )

    game.start(map_id = 2)

    game.init_bots_health(100)

    episodes = 100000

    last_states = []

    for _ in range(episodes):
        if game.is_player_dead():
            game.respawn_player()
        game.observe_state(params, last_states)
        action = np.random.randint(0, 29)
        game.make_action(action, frame_skip=1, sleep=None)
    game.close()
	
if __name__ == '__main__':
    main()
