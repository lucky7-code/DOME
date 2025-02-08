import random
import numpy as np
import torch
from model.trainer.fsl_trainer import FSLTrainer
from model.utils import (
    pprint, set_gpu,
    get_command_line_parser,
    postprocess_args,
)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    setup_seed(22)
    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    pprint(vars(args))
    set_gpu(args.gpu)
    trainer = FSLTrainer(args)
    # trainer.train()
    trainer.evaluate_test()
    trainer.final_record()
    print(args.save_path)
