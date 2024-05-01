
from gamingbench.prompts.observation_prompts import connect4
from gamingbench.prompts.observation_prompts import tictactoe
from gamingbench.prompts.observation_prompts import breakthrough
from gamingbench.prompts.observation_prompts import first_sealed_auction
from gamingbench.prompts.observation_prompts import liars_dice
from gamingbench.prompts.observation_prompts import negotiation
from gamingbench.prompts.observation_prompts import nim
from gamingbench.prompts.observation_prompts import pig
from gamingbench.prompts.observation_prompts import kuhn_poker
from gamingbench.prompts.observation_prompts import prisoners_dilemma


# maps
mapping = {
    'connect4': connect4,
    'tictactoe': tictactoe,
    'breakthrough': breakthrough,
    'first_sealed_auction': first_sealed_auction,
    'liars_dice': liars_dice,
    'negotiation': negotiation,
    'nim': nim,
    'pig': pig,
    'kuhn_poker': kuhn_poker,
    'python_iterated_prisoners_dilemma': prisoners_dilemma
}


def construct_observation_prompt(observations, environment_name):

    return mapping[environment_name].construct_observation_prompt(observations)

def construct_react_observation_prompt(observations, environment_name):

    return mapping[environment_name].construct_react_observation_prompt(observations)