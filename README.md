<img src="imgs/Jidi%20logo.png" width='300px'> 

# RLChina Competition - Gui Mao Winter Season 癸卯年冬赛季

This repo provide the source code for the [RLChina Competition - Gui Mao Winter Season](http://www.jidiai.cn/compete_detail?compete=44)



## Multi-Agent Game Evaluation Platform --- Jidi (及第)
Jidi supports online evaluation service for various games/simulators/environments/testbeds. Website: [www.jidiai.cn](www.jidiai.cn).

A tutorial on Jidi: [Tutorial](https://github.com/jidiai/ai_lib/blob/master/assets/Jidi%20tutorial.pdf)


## Environment
The competition uses an integrated version of [OvercookedAI games](https://github.com/HumanCompatibleAI/overcooked_ai)


### OvercookedAI-Integrated
<img src='https://jidi-images.oss-cn-beijing.aliyuncs.com/jidi/env103.gif' width=400>

- The integrated game contains three official maps, they are： 
  1. Cramped Room Tomato; 
  2. Forced Coordination Tomato; 
  3. Soup Coordination
- The game proceed by putting both agents sequentially in these three maps and ask them to prepare orders by cooperating with the other player. The ending state in a map is followed by an initial state in the next map and the agent observation will be marked *new_map=True*
- Each map will be run twice with agent index switched. For example in map one, player one controls agent one and player two controls agent two and they switch position and re-start the map when reaching an end. Thus, two players will play on three maps for six rounds in total.
- Each map last for 400 timesteps. The total episode length of the integrated game is 2400.
- Each agent observation global game state, the agent index and the new map indicator. The gloabl game state includes:
  - players: position and orientation of two characters and the held objects.
  - objects: the information and position of objects in the map.
  - bounus orders
  - all orders
  - timestep:  timestep in the current map
- Each agent has six action choices, they are:
  - Move Up
  - Move Down
  - Move Right
  - Move Left
  - Stay Still
  - Interact
- We use the default reward shaping, that is the value for each successful orders (x2 if bounus order). We will sum up all rewards in all maps as the score for one episode.


## Quick Start

You can use any tool to manage your python environment. Here, we use conda as an example.

```bash
conda create -n overcookai-venv python==3.7.5  #3.8, 3.9
conda activate overcookai-venv
```

Next, clone the repository and install the necessary dependencies:
```bash
git clone https://github.com/jidiai/Competition_OvercookedAI-2.git
cd Competition_OvercookedAI-2
pip install -r requirements.txt
```

Finally, run the game by executing:
```bash
python run_log.py
```

## Visualization

We provide two visualization method:
- Console: set `self.render_mode="console"` in `env/overcookedai_integrated.py` and the game map will be displayed as string on the console.
- Window: set `self.render_mode="window"` and the game map will be displayed in a seperated pygame window.


## Navigation

```
|-- Competition_OvercookedAI-2               
	|-- agents                              // Agents that act in the environment
	|	|-- random                      // A random agent demo
	|	|	|-- submission.py       // A ready-to-submit random agent file
	|-- env		                        // scripts for the environment
	|	|-- config.py                   // environment configuration file
	|	|-- overcookedai_integrated.py  // The environment wrapper		      
	|-- utils               
	|-- run_log.py		                // run the game with provided agents (same way we evaluate your submission in the backend server)
```



## How to test submission

- You can train your own agents using any framework you like as long as using the provided environment wrapper. 

- For your ready-to-submit agent, make sure you check it using the ``run_log.py`` scrips, which is exactly how we 
evaluate your submission.

- ``run_log.py`` takes agents from path `agents/` and run a game. For example:

>python run_log.py --my_ai "random" --opponent "random"

set both agents as a random policy and run a game.

- You can put your agents in the `agent/` folder and create a `submission.py` with a `my_controller` function 
in it. Then run the `run_log.py` to test:

>python run_log.py --my_ai your_agent_name --opponent xxx

- If you pass the test, then you can submit it to the Jidi platform. You can make multiple submission and the previous submission will
be overwritten.


