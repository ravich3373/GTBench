
import itertools
import re

from gamingbench.agents.base_agent import BaseAgent
from gamingbench.prompts.system_prompts import construct_system_prompt
from gamingbench.prompts.observation_prompts import construct_observation_prompt
from gamingbench.prompts.step_prompts.alphallm_agent import construct_step_prompt, construct_voting_prompt

class AlphaLLMAgent(BaseAgent):

    def __init__(self, config, **kwargs):
        super(AlphaLLMAgent, self).__init__(config)
        self.task_steps = config.task_steps
        self.method_generate = config.method_generate
        self.method_evaluate = config.method_evaluate
        self.method_select = config.method_select
        self.n_generate_sample = config.n_generate_sample
        self.n_evaluate_sample = config.n_evaluate_sample
        self.n_select_sample = config.n_select_sample
        self.prompt_sample = config.prompt_sample
        
        self.learnings = ""
        self.prev_obs = ""
        self.infer_graph = ""
        self.opp_next_pred = ""

    def conclude(self, observations, status):
        self.logger.info('-' * 20 + 'AlphaLLMAgent Begin Conclusion' + '-' * 20)
        env_name = observations['env_name']
        system_prompt = construct_system_prompt(env_name)
        observation_prompt = construct_observation_prompt(observations, environment_name=env_name)
        step_instruct = construct_step_prompt(observations)
        step_prompt = step_instruct['prompt']
        step_regex = step_instruct['regex']
        stop_signs = step_instruct['stop_signs']

        query_list = []
        ys = [""]
        x = self.construct_init_messages(system_prompt, observation_prompt + '\n' + step_prompt)
        # learning
        # ravi
        opponent_move = ""
        if len(observations["opponent_moves"]) > 0:
            opponent_move = observations['opponent_moves'][-1]
        # ravi
        if self.infer_graph and opponent_move != self.opp_next_pred:
            learn_x = x[-1]['content'] = '\n' + "Previous game state:\n" + self.prev_obs + "\n" 

            learn_x += "Game Knowledge Gained:\n" + self.learnings + "\n"
            learn_x += "Game Progression Expected by LLM including opponent's Moves:\n" + self.infer_graph + f"\nActual Move of opponent:\n{opponent_move}\n"
            if status == "win":
                learn_x += "You won the game. "
            elif status == "loss":
                learn_x += "The opponent won the game. "
            else:  # draw
                learn_x += "The game is a draw. "

            learn_x += "Extract concise knowledge from this game that you can use to improve yourself. Combine this knowledge with the already gained knowledge from 'Game Knowledge Gained:' section and output. "
            learn_x += "This knowledge should strictly follow the output format:\n New Knowledge Begin:\n write knowledge gained here. \nNew Knowledge End\n"
            
            x[-1]["content"] = learn_x
            new_ks = [self._get_samples(x, y, self.n_generate_sample, stop=None) for y in ys]
            query_list += [query[1] for query in new_ks]
            new_ks = [new_k[0] for new_k in new_ks]
            # extract knowledge
            new_k = new_ks[0][0].split("\n")[1]
            if len(new_k) > 0:
                self.learnings = new_k.strip()
        
        return query_list
        
    
    def step(self, observations):
        self.logger.info('-' * 20 + 'AlphaLLMAgent Begin' + '-' * 20)
        # we follow the official tot implementation: https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/src/tot/methods/bfs.py
        env_name = observations['env_name']
        system_prompt = construct_system_prompt(env_name)
        observation_prompt = construct_observation_prompt(observations, environment_name=env_name)

        step_instruct = construct_step_prompt(observations)
        step_prompt = step_instruct['prompt']
        step_regex = step_instruct['regex']
        stop_signs = step_instruct['stop_signs']

        voting_instruct = construct_voting_prompt(observations)
        voting_prompt = voting_instruct['prompt']
        voting_regex = voting_instruct['regex']

        ys = ['']
        query_list = []
        self.task_steps = 1  # ravi
        
        for step in range(self.task_steps):
            x = self.construct_init_messages(system_prompt, observation_prompt + '\n' + step_prompt)
            # learning
            # ravi
            opponent_move = ""
            if len(observations["opponent_moves"]) > 0:
                opponent_move = observations['opponent_moves'][-1]
            # ravi
            if self.infer_graph and opponent_move != self.opp_next_pred:
                learn_x = x[-1]['content'] = '\n' + "Previous game state:\n" + self.prev_obs + "\n" 

                learn_x += "Game Knowledge Gained:\n" + self.learnings + "\n"
                learn_x += "Game Progression Expected by LLM including opponent's Moves:\n" + self.infer_graph + f"\nActual Move of opponent:\n{opponent_move}\n"
                learn_x += "Analyze the opponent's move assuming he is very good at the game and has a strategy to win. "
                learn_x += "Extract concise knowledge from this move made by the opponent that you can use to improve yourself. Combine this knowledge with the already gained knowledge from 'Game Knowledge Gained:' section and output. "
                learn_x += "This knowledge should strictly follow the output format:\n New Knowledge Begin:\n write knowledge gained here. \nNew Knowledge End\n"
                
                x[-1]["content"] = learn_x
                new_ks = [self._get_samples(x, y, self.n_generate_sample, stop=stop_signs[step]) for y in ys]
                query_list += [query[1] for query in new_ks]
                new_ks = [new_k[0] for new_k in new_ks]
                # extract knowledge
                new_k = new_ks[0][0].split("\n")[1]
                if len(new_k) > 0:
                    self.learnings = new_k.strip()
            # generation
            x = self.construct_init_messages(system_prompt, observation_prompt + '\n' + "Game Knowledge Gained:\n" + self.learnings + '\n' + "Your Turn:\n" + step_prompt)
            #x[-1]['content'] += '\n' + "Game Knowledge Gained:\n" + self.learnings + "\n"
            if self.method_generate == 'sample':
                new_ys = [self._get_samples(x, y, self.n_generate_sample, stop=stop_signs[step]) for y in ys]
                query_list += [query[1] for query in new_ys]
                new_ys = [new_y[0] for new_y in new_ys]

                # opponent predicted next action
                opp_next_pred = re.findall(r"Opponent Best Action:\n<C[1-3]R[1-3]>", new_ys[0][0])
                if len(opp_next_pred) > 0:
                    self.opp_next_pred = opp_next_pred[0].split("\n")[1].strip()
                self.infer_graph = new_ys[0][0]
                self.prev_obs = observation_prompt
            else:
                raise NotImplementedError
            new_ys = list(itertools.chain(*new_ys))
            ids = list(range(len(new_ys)))
            # evaluation
            # x = self.construct_init_messages(system_prompt, observation_prompt)
            # if self.method_evaluate == 'vote':
            #     values, query = self._vote(x, new_ys, self.n_evaluate_sample, voting_prompt, voting_regex)
            #     query_list.append(query)
            # else:
            #     raise NotImplementedError

            # selection
            # if self.method_select == 'greedy':
            #     select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:self.n_select_sample]
            # else:
            #     raise NotImplementedError

            #ys = [new_ys[select_id] for select_id in select_ids]
            ys = new_ys

        parsed_moves = self.parse_with_regex(ys, step_regex)
        parsed_moves = self.post_processing(parsed_moves, majority_vote=True)
        self.logger.info('-' * 20 + 'ToTAgent End' + '-' * 20)
        return parsed_moves, query_list

    def _get_samples(self, messages, y, n_generate_sample, stop):
        messages[-1]['content'] += '\n' + y
        self.logger.info('Thought/Action Prompt:')
        self.logger.info(messages[-1]['content'])
        responses, query = self.llm_query(messages, n=n_generate_sample, stop=stop, prompt_type='plan')
        self.logger.info('Thought/Action Response:')
        self.logger.info(responses)
        return responses, query


    def _vote(self, messages, y, n_evaluation_sample, voting_prompt, voting_regex):
        values = [0] * len(y)
        for idx, gen in enumerate(y):
            messages[-1]['content'] += '\n' + f'Choice{idx + 1}: {gen}'
        messages[-1]['content'] += '\n' + voting_prompt
        self.logger.info('Voting Prompt:')
        self.logger.info(messages[-1]['content'])
        responses, query = self.llm_query(messages, n=n_evaluation_sample, stop=None, prompt_type='vote')
        self.logger.info('Voting Response:')
        self.logger.info(responses)
        votes = self.parse_with_regex(responses, regex=voting_regex)
        filtered_votes = []
        for r in votes:
            # Use the last matched item as the model answer
            r = r[-1]
            if r is not None and int(r) - 1 in list(range(len(y))):
                filtered_votes.append(int(r) - 1)
            else:
                # TODO error print
                pass

        for v in filtered_votes:
            values[v] += 1

        return values, query
