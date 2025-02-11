class Settings:
    def __init__(self):
        self.phase = None
        self.context_path = None
        self.auth_path = None
        self.threshold = None
        self.window = None
        self.rbound = None
        self.lbound = None
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.df_auth = None
        self.plot_mode = False
        self.verbose = False
        self.falcon_verbose = False
        self.debug = False
        self.samples = {}
        self.samples_train = {}
        self.samples_test = {}
        self.samples_val = {}
        self.samples_auth = {}
        self.execution_path = None
        self.n_intervals = None
        self.model = None
        
    def update(self, updates):
        for key, value in updates.items():
            if hasattr(self, key): 
                setattr(self, key, value)
            else:
                print(f"Warning: Setting '{key}' not found.") 

settings = Settings() 