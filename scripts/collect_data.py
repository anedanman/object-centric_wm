from cli.collect import CollectArgs
from data_collection.collect import collect

if __name__ == "__main__":
    args = CollectArgs().parse_args()
    collect(args)
