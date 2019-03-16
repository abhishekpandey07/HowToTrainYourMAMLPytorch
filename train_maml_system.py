from data import MetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder
from few_shot_learning_system import MAMLFewShotClassifier
from utils.parser_utils import get_args
from utils.dataset_tools import maybe_unzip_dataset
import ptvsd
# Combines the arguments, model, data and experiment builders to run an experiment
args, device = get_args()
# Allow other computers to attach to ptvsd at this IP address and port.
ptvsd.enable_attach(address=('0.0.0.0', 5678), redirect_output=True)
# Pause the program until a remote debugger is attached
ptvsd.wait_for_attach()
model = MAMLFewShotClassifier(args=args, device=device,
                             data_shape=(2, args.input_channels,args.input_width))
# breakpoint()
maybe_unzip_dataset(args=args)
data = MetaLearningSystemDataLoader
maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)
maml_system.run_experiment()
