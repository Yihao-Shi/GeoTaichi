import os
import glob
import t2v_metrics
import json

prompts = [
    'a cozy cartoon setup with a bed, beside table, lamp, bookshelf, and a small dog on the bed.',
    'a luxurious CG scene with a couch, coffee table, small art piece, a cozy rug.',
    'a warm setup with a bed, bedside tables, plant, a few books, and a small decorative lamp.',
    'a baby bunny sitting on top of a stack of pancakes',
    'a Wizard standing in front of a Wooden Desk, gazing into a Crystal Ball perched atop the Wooden Desk, with a Stack of Ancient Spell Books perched atop the Wooden Desk',
    'a blue jay standing on a large basket of rainbow macarons',
    'On a table, there is a vase with a bouquet of flowers. Beside it, there is a plate of cake',
    'Horizontal perspective, showing the complete objects. In a basket of fruit, there are 5 fruits.',
    'Horizontal perspective, showing the complete objects. Four stacked cups and four stacked plates.',
    'A white ceramic cup holds two blue and green toothbrushes, a blue and white toothpaste tube, and a blue-handled razor, while a white soap dish beside it contains two bars of soap—one white and one beige.',
    'A vintage wooden radio with a small cow figurine on top sits on a stack of three hardcover books, next to a wooden cup holding colorful pencils.',
    'A brown leather sofa decorated with plush toys, including a large teddy bear, a gray elephant, a white rabbit, a yellow giraffe, and two throw pillows, sits in a cozy room with two round burgundy floor cushions in front.',
    'A round blue inflatable pool filled with small blue, white, and turquoise plastic balls features a toy sailboat, two yellow rubber ducks, a green turtle, and an orange starfish.',
    'A wooden table with a red fire extinguisher sits in front of a metal shelving unit with a gray cap on one of the shelves, flanked by large cardboard sheets on one side and a wheeled cart holding potted green plants on the other.',
    'A stack of colorful wooden blocks arranged vertically, featuring red, blue, yellow, green, orange, and purple pieces, balanced on a flat surface.',
    'a cute plush toy dog with five different colored hats stacked on its head'
]


def compute_metrics_one_object(score_model, obj_folder, prompt):
    all_renders = glob.glob(os.path.join(obj_folder, 'render*.png')) \
        + glob.glob(os.path.join(obj_folder, '*', 'render*.png'))

    scores = score_model(all_renders, [prompt])
    # print(scores)
    mean_score = sum(scores) / len(scores)
    return mean_score



if __name__ == "__main__":
    clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl')

    source_dir = '/data1/users/yuanhao/guying_proj/eval/baselines_renders_2k'
    # baseline_names = os.listdir(source_dir)
    baseline_names = ['blenderMCP', 'graphdreamer_all', 'midi']
    for baseline_name in baseline_names:
        source_folder = os.path.join(source_dir, baseline_name)
        # get all the folders in the source folder
        objs = os.listdir(source_folder) 
        objs = [a for a in objs if os.path.isdir(os.path.join(source_folder, a))]
        obj_ids = [int(a.split('_')[0]) for a in objs]
        scores = []
        scores_dict = {}
        for obj, obj_id in zip(objs, obj_ids):
            obj_folder = os.path.join(source_folder, obj)
            prompt = prompts[obj_id-1]
            score = compute_metrics_one_object(clip_flant5_score, obj_folder, prompt)
            scores.append(score)
            scores_dict[obj] = score.item()
            # print(f'Object {obj} with prompt {prompt} has score {score}')

        mean_score = sum(scores) / len(scores)
        # print()
        print(f'Baseline {baseline_name} has mean score {mean_score.item()}')

        # save the scores to a json file
        save_path = os.path.join(source_folder, 'scores.json')
        with open(save_path, 'w') as f:
            json.dump(scores_dict, f, indent=4)
        print(f'Saved scores to {save_path}')