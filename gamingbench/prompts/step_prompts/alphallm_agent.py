
from gamingbench.prompts.regex_and_format import get_step_env_regex_and_format

def _get_stop_signs(env_name):
    stop_signs = [None, None]
    return stop_signs


def construct_step_prompt(observation):

    env_name = observation.get('env_name', '')

    regex, format = get_step_env_regex_and_format(env_name)

    stop_signs = _get_stop_signs(env_name)

    prompt = f"""1)First think about your current situation, 2) then you must choose one action from legal actions to set up advantages.

3) Identify the valid moves for your opponent after you make your move. Remove the move you made from the moves available to you to answer this.

4) Then, you should identify what you think is the best move for your opponent(from opponent's perspective) so that he gains advantage over you.

5) Finally Analyze If the move you chose in step 2 is still the best move considering your oppoenet's future moves. If not change your move. Answer without any explanation.
    
Your output should be of the following format strictly:

Initial Thought:
Your thought.
    
Initial Best Action:
Your action wrapped with <>, e.g., {format}

Valid Opponent Moves after your move:
Valid moves available for opponent, wrapped with <>, e.g., <C1R1>, ... , <C3R2>.

Opponent Thought:
Your thought from opponent's perspective.

Opponent Best Action:
Opponent Action wrapped with <>, e.g., {format}

Thought:
Your final thought.

Action:
Your action wrapped with <>, e.g., {format}
"""

    return {
        'prompt': prompt,
        'regex': regex,
        'stop_signs': stop_signs
    }


def construct_voting_prompt(observation):
    prompt = '''Given an instruction and several choices, decide which choice is most promising. Analyze each choice in detail, then conclude in the last line "The best choice is {s}", where s the integer id of the choice.'''
    return {
        'prompt': prompt,
        'regex': '.*best choice is .*(\d+).*'
    }
