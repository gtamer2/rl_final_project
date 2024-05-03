import llm_blender

blender = llm_blender.Blender()
blender.loadranker("llm-blender/PairRM") # load PairRM

def getScores(inputs, candidates, ref_candidates, batch_size=32):
    rewards = blender.rank_with_ref(inputs, candidates, return_scores=True, batch_size=batch_size, ref_candidates=ref_candidates) 
    return rewards
  
# Usage
# rewards = getScores(['How are you doing?'], [['I am doing fine']], ['I am fine'])