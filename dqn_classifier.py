import os
import numpy as np

class TextWorld(object):
    def __init__(self,path, categories):
        self.path = path
        self.categories = categories
        self.idx_to_cat = {}
        self.cat_to_idx = {}
        self.reader = self._read()
        self.vec = self._vectorize()
        self.vocab_size = len(self.vec.vocabulary_)
        self.old_state = None

        for i, cat in enumerate(categories):
            self.idx_to_cat[i+1] = cat
            self.cat_to_idx[cat] = i+1

    def _vectorize(self):       
        text = []
        for category in self.categories:
            sub_dir = os.path.join(self.path,category)
            for fname in os.listdir(sub_dir):
                full_fname = os.path.join(sub_dir,fname)
                with open(full_fname, 'r') as f:
                    for line in f:
                        text.append(line)
                        
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(stop_words='english',  # Remove common words
                                    token_pattern=r'[a-zA-Z]{3,}',  # regex to choose the words
                                    lowercase=True,
                                    max_features=10000,
                                    use_idf=True
                                    ) 
        vec.fit(text) # Fit the vectorizer, but not transform!
        return vec
        

    ## Generator for the state
    def _read(self):
        for category in self.categories:
            sub_dir = os.path.join(self.path,category)
            for fname in os.listdir(sub_dir):
                full_fname = os.path.join(sub_dir,fname)
                with open(full_fname, 'r') as f:
                    for line in f:
                        x = self.vec.transform([line])
                        y = self.cat_to_idx[category]
                        yield x,y
    

    def step(self, action):
        """
        :param action: Action to be performed. 0 -> read, action > 0 -> classify
        :return: tuple of {state, reward, done}.
        """
        category = 0
        if action == 0:
            # Read a line
            eof = False
            line, category = next(self.reader)
            if line is None:
                eof = True
                line = self.old_state
            else:
                self.old_state = line

            return line, 0, eof
        else:
            # Classify
            self.reader.close()
            if action == category:
                return None, 1, True
            else:
                return None, -1, True

    def reset(self):
        self.reader = self._read()
        state, _ , _ = self.step(0)
        return state
        



class MLP:
    def __init__(self,env,n_actions):
        self.n_actions = n_actions    
        #create agent
        from sklearn.neural_network import MLPClassifier
        self.agent = MLPClassifier(hidden_layer_sizes=(20,20),
                        activation='tanh',
                        warm_start=True, #keep progress between .fit(...) calls
                        max_iter=1 #make only 1 iteration on each .fit(...)
                        )
        #initialize agent to the dimension of state times number of actions
        state = env.reset().todense().reshape(1,-1)
        a = np.array([0]).reshape(1,-1)
        sa = np.append(state,a,axis=1)
        self.agent.fit(sa,np.array([0], dtype=np.float))
    
    def predict(self,state):
        state = state.todense().reshape(1,-1)
        all_a = [np.array([a]).reshape(1,-1) for a in range(self.n_actions+1)]
        sa_pairs = [np.append(state,a, axis=1) for a in all_a]
        return [self.agent.predict(sa) for sa in sa_pairs ]
    
    def fit(self,sa,td_target):
        return self.agent.fit([sa,td_target])


        
def make_policy(estimator, epsilon, actions):
    def policy_fn(state):
        prescribed_action = np.argmax(estimator.predict(state))
        if np.random.rand()>epsilon:
            action = prescribed_action 
        else:
            action = np.random.choice(actions)
        return action
    return policy_fn

           
if __name__ == "__main__":
    categories = ['soc.religion.christian','rec.sport.hockey']
    dir_name = "20_newsgroup_small"
    
    # Initialize environment
    env = TextWorld(dir_name, categories)
    
    # Launch estimator and policy 
    mlp = MLP(env,3)
    policy = make_policy(mlp,0.1,[0,1,2])
    
    # training params
    gamma = 0.9
    n_episodes = 1000

    # Episode
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = policy(state)
            state, reward, done = env.step(action)
            ep_reward += reward

            if done:
                td_target = reward
            else: 
                td_target = reward + gamma*np.max(mlp.predict(state))
        
        if state is not None:
            s = state.todense().reshape(1,-1)
            a = np.array([action]).reshape(1,-1)
            sa = np.append(s,a,axis=1)
            mlp.fit(sa,td_target)            
        print("Episode: {}. Episode reward: {}".format(ep,ep_reward))
