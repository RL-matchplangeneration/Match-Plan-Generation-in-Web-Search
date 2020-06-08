# Match Plan Generation in Web Search with Parameterized Action Reinforcement Learning

## About Environment Installation
- Match plan generation
  
    - Microsoft internal access
    
- Benchmark environments
    - referring to `https://github.com/cycraig/MP-DQN`
        - ```bash
          pip install -e git+https://github.com/cycraig/gym-platform#egg=gym_platform
          pip install -e git+https://github.com/cycraig/gym-goal#egg=gym_goal
          ```

## Code Running

- ```bash
  pip install -r requirements.txt
  git clone https://github.com/Microsoft/nni.git
  ```

- Running

  - run_benchmark.py: manually tune hyper-parameters
  - run_benchmark_nni.py: use NNI to tune hyper-parameters

- Parameters: 

  - referring to the code run_xxx.py
  - loading checkpoint: see run_benchmark.py dict:debug,  load_filename = 'Goal_psac3_dl10_d100_320000_May_31_2020_16_26_12' (example)

