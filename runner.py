import argparse

# from AC_ import A3CTrainer
# from AC_ import A2CTrainer
# from AC_ import actor_critic_inference
# from AC_ import evaluate_actor_critic
# from Actor_Critic import A3CTrainer
# from Actor_Critic import A2CTrainer
# from Actor_Critic import actor_critic_inference
# from Actor_Critic import evaluate_actor_critic
# from DQN_base import DQNTrainer
# from DQN_base import dqn_inference
# from DQN_base import evaluate_dqn
from DQN import DQNTrainer
from DQN import dqn_inference
from DQN import evaluate_dqn
from A2C import A2CTrainer
from A2C import a2c_inference
from A2C import evaluate_a2c
from params import Params

# RGB 210, 160, 3 k frame uniformly sampled {2, 3, 4}

## https://github.com/AppliedDataSciencePartners/WorldModels
## https://github.com/AGiannoutsos/car_racer_gym
## https://github.com/ctallec/world-models

def get_trainer(model_type, params):
    model_path = 'models/' + model_type + '.pt'
    if model_type == 'a2c':
        return A2CTrainer(params, model_path)
    # elif model_type == 'a3c':
    #     return A3CTrainer(params, model_path)
    elif model_type == 'dqn':
        return DQNTrainer(params, model_path)
    return None

def run_training(model_type):
    params = Params('params/' + model_type + '.json')
    trainer = get_trainer(model_type, params)
    trainer.run()

def run_inference(model_type):
    params = Params('params/' + model_type + '.json')
    if model_type == 'dqn':
        score = dqn_inference('models/' + model_type + '.pt')
    else:
        score = a2c_inference('models/' + model_type + '.pt')

    print('Total score: {0:.2f}'.format(score))

def run_evaluation(model_type):
    params = Params('params/' + model_type + '.json')
    if model_type == 'dqn':
        score = evaluate_dqn('models/' + model_type + '.pt')
    else:
        score = evaluate_a2c('models/' + model_type + '.pt')

    print('Average reward after 100 episodes: {0:.2f}'.format(score))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True,
                        choices=['a2c', 'a3c', 'dqn'],
                        help='Which model to run / train.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true',
                       help='Train model.')
    group.add_argument('--inference', action='store_true',
                       help='Model inference.')
    group.add_argument('--evaluate', action='store_true',
                       help='Evaluate model on 100 episodes.')

    args = vars(parser.parse_args())
    if args['train']:
        run_training(args['model'])
    elif args['evaluate']:
        run_evaluation(args['model'])
    else:
        run_inference(args['model'])