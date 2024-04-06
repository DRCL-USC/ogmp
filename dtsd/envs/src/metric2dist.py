import numpy as np

# a class that takes in a metric and returns a probability distribution
class metric2dist:
    def __init__(
                    self,
                    **kwargs

                ):
        
        self.conf = kwargs

        

        if self.conf['n_cases'] is 'infinite':
            # continuous distribution
            pass
        else:
            # discrete distribution
            self.metric_record = np.zeros(self.conf['n_cases'])

        # by default set  0 gamma record
        if 'update_metric_type' not in self.conf.keys():
            self.conf['update_metric_type'] = 'telescopic_gamma' 
            self.conf['gamma'] = 0.0

        if 'sample_type' not in self.conf.keys():
            self.conf['sample_type'] = 'uniform'
        elif self.conf['sample_type'] in ['eps_norm','adaptive'] and 'epsilon' not in self.conf.keys():
            # set default epsilon
            self.conf['epsilon'] = 0.0

        # print(self.conf)

    def update_metric(self,metric,case_id):
        getattr(self,'um_'+self.conf['update_metric_type'])(metric,case_id)

    def return_prob(self):
        return getattr(self,'cd_'+self.conf['sample_type'])()
    
    # update metric functions
    def um_telescopic_gamma(self, metric, case_id):
        self.metric_record[case_id] = self.conf['gamma']*self.metric_record[case_id] + metric
    
    # cd ditribution functions
    def cd_uniform(self):
        return np.ones(self.conf['n_cases'])/self.conf['n_cases']
    
    def cd_adaptive_norm(self, metric_ext = None):

        if metric_ext is None:
            # use internal metric
            metric_ext = self.metric_record


        min_metric = np.min(metric_ext)
        max_metric = np.max(metric_ext)

        if min_metric == max_metric:
            # all cases are the same
            norm_metric = np.ones(self.conf['n_cases'])/self.conf['n_cases']
        else:
            # normalise
            norm_metric   = (metric_ext - min_metric)/(max_metric - min_metric)

        # convert to probability
        norm_metric += self.conf['epsilon']
        norm_metric /= np.sum(norm_metric)
        return norm_metric    
    
    def cd_adaptive_norm_inv(self):

        neg_metric = -self.metric_record
        return self.cd_adaptive_norm(neg_metric)

    def cd_adaptive_softmax(self, metric_ext = None):

        if metric_ext is None:
            # use internal metric
            metric_ext = self.metric_record

        # softmax
        exp_metric = np.exp(metric_ext)
        exp_metric /= np.sum(exp_metric)
        return exp_metric

    def cd_adaptive_softmax_inv(self):
        neg_metric = -self.metric_record
        return self.cd_adaptive_softmax(neg_metric)

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import os
    from tqdm import tqdm
    N_CASES = 5
    N_POLICY_UPDATES = 1000
    SAMPLE_TYPE = 'adaptive_norm_inv'
    # GSP: GRID SEARCH PARAMS 
    # [start,stop,n_samples]
    EPSILON_GSP = {
                    'start':0.0,
                    'stop':1.0,
                    'num':1
                    }
    GAMMA_GSP = {
                    'start':0.0,
                    'stop':1.0,
                    'num':1
                }
    POLICY_FORGET_PROB_GSP = {
                                'start':0.0,
                                'stop':1.0,
                                'num':10
                            }

    PLOT_PROB_EVOLUTION = False
    VERBOSE = False

    PATH2PLOTS = "dtsd/analysis_results/adapt_sample_trng_abalation/"+SAMPLE_TYPE+"/"
    os.makedirs(PATH2PLOTS,exist_ok=True)

    # mimc an assymptotically improving agent
    def dummy_agent(case_id,forget_prob=0.1):
        # mimics a perfect learner if forget_prob = 0, else forgets with prob. forget_prob
        # assumes the policy learns the given case id 
        score_record[case_id] += np.random.random()

        if forget_prob != 0 and np.random.random() < forget_prob:
            # cannot forget the case you are learning right now
            forget_case_id = case_id
            
            while forget_case_id == case_id:
                forget_case_id = np.random.choice(range(N_CASES))
            score_record[forget_case_id] -= np.random.random()

        return score_record[case_id]

    
    
    
    hyperparam_id = 0
    mean_trained_scores = []
    std_trained_scores = []
    hyperaparams = []

    pbar = tqdm(
                total=GAMMA_GSP['num']*EPSILON_GSP['num']*POLICY_FORGET_PROB_GSP['num'],
                leave=False
                )
    
    for EPSILON in np.linspace(**EPSILON_GSP).round(2):
        for GAMMA in np.linspace(**GAMMA_GSP).round(2):
            for POLICY_FORGET_PROB in np.linspace(**POLICY_FORGET_PROB_GSP).round(2):
                pbar.update(1)
                sampler = metric2dist(
                                    sample_type = SAMPLE_TYPE,
                                    n_cases = N_CASES,
                                    gamma = GAMMA,
                                    epsilon = EPSILON,
                                    )

                # original score record
                score_record = np.zeros(N_CASES)    

                
                perf_case = [[] for i in range(N_CASES)]
                score_that_epoch = []
                for i in range(N_POLICY_UPDATES):

                    # prob = sampler.cd_uniform()
                    prob = sampler.return_prob()
                    # prob = sampler.cd_eps_norm()
                    
                    if VERBOSE:
                        print("prob:", prob.round(2),"prob_sum:",np.sum(prob).round(2))
                    
                    case_id = np.random.choice(range(N_CASES),p=prob)
                    score = dummy_agent(case_id,forget_prob=POLICY_FORGET_PROB)
                    
                    score_that_epoch.append(score)
                    sampler.update_metric(score,case_id)
                    
                    for ci, sc in enumerate(score_record):
                        perf_case[ci].append(sc)



                    # perf_case[case_id].append(score)
                    if VERBOSE:
                        print('#############################')
                        print("case_id: ",case_id, "score: ",round(score,2), )
                        print("metric true: ",score_record.round(2))
                        print("metric_record: ",sampler.metric_record.round(2))


                    if PLOT_PROB_EVOLUTION:
                        plt.bar(range(N_CASES),prob)
                        plt.pause(0.024)
                        plt.clf()
                
                if PLOT_PROB_EVOLUTION:
                    plt.tight_layout()
                    plt.xlabel("case_id")
                    plt.ylabel("prob")
                    # plt.show()
                    plt.close()

                # figure 2
                plt.title("performance over modes eps:{} gamma:{} forget_prob:{}".format(EPSILON,GAMMA,POLICY_FORGET_PROB))
                for i in range(N_CASES):
                    plt.plot(perf_case[i],label='case_'+str(i),color='C'+str(i))
                plt.tight_layout()
                plt.grid()
                plt.savefig(PATH2PLOTS+"/pom_"+str(hyperparam_id)+".png")
                # plt.show()
                plt.close()


                # figure 3
                plt.title("score that epis eps:{} gamma:{} forget_prob:{}".format(EPSILON,GAMMA,POLICY_FORGET_PROB))
                plt.plot(
                            score_that_epoch,
                            # label='c'+str(i),
                            color='C0',
                        )
                plt.tight_layout()
                plt.grid()
                plt.savefig(PATH2PLOTS+"/ste_"+str(hyperparam_id)+".png")
                # plt.show()
                plt.close()



                mean_trained_score = np.mean(score_record)
                std_trained_score = np.std(score_record) 

                mean_trained_scores.append(mean_trained_score)
                std_trained_scores.append(std_trained_score)
                hyperaparams.append([EPSILON,GAMMA,POLICY_FORGET_PROB])
                # print(
                #         hyperparam_id,
                #         "eps: ",EPSILON,
                #         "gamma: ",GAMMA,
                #         "forget_prob: ",POLICY_FORGET_PROB,
                #         "mts: ",mean_trained_score.round(2),
                #         "sts: ",std_trained_score.round(2)
                #     )
                # print("saved at: ",PATH2PLOTS+"/"+str(hyperparam_id)+".png")
                hyperparam_id += 1
    
    # mean_trained_scores = np.array(mean_trained_scores)
    # std_trained_scores = np.array(std_trained_scores)
    # hyperaparams = hyperaparams


    norm_mu_scores = mean_trained_scores/np.max(mean_trained_scores)
    norm_sd_scores = std_trained_scores/np.max(std_trained_scores)

    # max mu, min sd
    norm_min_plus_sd = norm_mu_scores - norm_sd_scores


    mu_max_ids = np.argsort(norm_min_plus_sd)
    # sd_min_ids = np.argsort(std_trained_scores)


    top_n = 5
    print("\tbest mu score candidates:")
    for rank in range(top_n):
        id = mu_max_ids[-1-rank]
        print(  '\t\t',
                rank+1,
                ' mu:',mean_trained_scores[id].round(2),
                ' exp:',hyperaparams[id],
                ' sd:',std_trained_scores[id].round(2)
                )

    # print("\tbest sd score candidates:")
    # for rank in range(top_n):
    #     id = sd_min_ids[rank]
    #     print(  '\t\t',
    #             rank+1,
    #             ' mu:',mean_trained_scores[id].round(2),
    #             ' exp:',hyperaparams[id],
    #             ' sd:',std_trained_scores[id].round(2)
    #             )
