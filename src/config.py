import json


class Config(object):
    def __init__(self, args=None, file=None):
        # search config
        self.topk = 5
        self.langSearch = 'en'
        self.langQuestion = 'en'
        self.dataset = 'dev'
        self.search_params = {"k1": 0.9203, "b": 0.2173}

        # connection config
        self.server = {"host": 'localhost', 'port': 22221}
        self.stop = True

        # training config
        self.lr = 1e-5
        self.params = "/path/to/parameters"
        self.freeze_encoder = True
        self.eval_dataset = "dev"

        # general config
        self.test = 1
        self.dry_run = False
        self.dump_config = None

        self.arg_params = []
        self.cfg_params = []

        # parse arguments
        if file is not None:
            self.load(file)
        if hasattr(args, 'config'):
            self.load(args.config)
        if args is not None:
            self.parse_args(args)

        # if dump file passed by param dump configuration
        if self.dump_config is not None:
            print(self.dump_config)
            self.dump(self.dump_config)

    def __str__(self):
        str = ""
        for k, v in vars(self).items():
            if type(v) == dict:
                for kk, vv in v.items():
                    str += "{0} ({1}): {2}\n".format(kk.ljust(14), self.val_from(kk), vv)
            if k in self.arg_params:
                str += "{0} ({1}): {2}\n".format(k.ljust(14), self.val_from(k), v)
        return str

    def val_from(self, key):
        if key in self.arg_params:
            val_from = "arg"
        elif key in self.cfg_params:
            val_from = "cfg"
        else:
            val_from = "def"
        return val_from

    def parse_args(self, args):
        if args is None:
            return
        config = vars(args)
        self.parse(config, self.arg_params)

    def load(self, file):
        if file is None:
            return
        with open(file) as fp:
            config = json.load(fp)
            self.parse(config, self.cfg_params)

    def parse(self, cfg: dict, option_list):
        for k, v in cfg.items():
            if v is None:
                continue
            option_list.append(k)
            if k == 'k1':
                self.search_params['k1'] = v
            elif k == 'b':
                self.search_params['b'] = v
            elif k == 'server':
                self.server['host'] = v
            elif k == 'port':
                self.server['port'] = v
            else:
                setattr(self, k, v)

    def dump(self, file=None):

        if file is None or self.dump_config is None:
            raise RuntimeError("Config could not be dumped, file not specified")

        file = self.dump_config if file is None else file

        print("Dumping config to: {}".format(file))
        cfg = vars(self)
        dump = {}
        do_not_include = ['test', 'dry_run', 'dump_config', 'cfg_params', 'arg_params', 'config', 'save_as']
        for k, v in cfg.items():
            if k in do_not_include:
                continue
            if type(v) == dict:
                for kk, vv in v.items():
                    dump[kk] = vv
            else:
                dump[k] = v

        with open(file, mode='w+') as fp:
            json.dump(dump, fp, indent=4)
