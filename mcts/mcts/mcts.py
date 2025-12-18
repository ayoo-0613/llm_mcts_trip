# adapted from https://github.com/jys5609/MC-LAVE-RL.git

import numpy as np
from tqdm import tqdm
import mcts.mcts.utils as utils
from collections import defaultdict

DISCOUNT_FACTOR = 0.95


class StateNode:
    def __init__(self, reward=0, done=False):
        self.ob = None
        self.look = None
        self.inv = None
        self.state = None
        self.prev_action = None
        self.id = None
        self.valid_actions = None
        self.history = []
        self.parent = None
        self.parent_action_id = None
        self.best_action_node = None

        self.N = 0
        self.children = []
        self.children_probs = []
        self.children_prob_pool = []
        self.remaining_action_indices = []
        self.reward = reward / (1 - DISCOUNT_FACTOR)
        self.score = 0
        self.done = done
        self.predicted_reward = 0
        self.use_llm = False


class ActionNode:
    def __init__(self, action):
        self.action = action
        self.N = 0
        self.Q = 0
        self.Q_hat = 0
        self.Rs = []
        self.children = None
        self.children_id = None


class MCTSAgent:
    def __init__(
        self,
        args,
        env,
        policy=None,
        name="MCTS",
        uct_type="PUCT",
        valid_action_dict=None,
        actions_info=None,
        log_dir=None,
        visited_transitions=None,
        replay_file=None,
        use_llm=True,
    ):
        self.env = env
        self.name = name
        self.best_action_node = None
        self.uct_type = uct_type
        self.seed = getattr(args, "seed", 0)
        self.round = getattr(args, "round", 0)
        self.root = None
        self.debug = getattr(args, "debug", False)

        self.exploration_constant = getattr(args, "exploration_constant", 1)
        self.bonus_constant = getattr(args, "bonus_constant", 0)
        self.max_depth = getattr(args, "max_depth", 0)
        self.simulation_per_act = getattr(args, "simulation_per_act", 1)
        self.discount_factor = getattr(args, "discount_factor", DISCOUNT_FACTOR)
        self.visited_transitions = visited_transitions

        self.action_selection_temp = 0.1 / (self.round + 1)

        self.policy = policy
        self.actions = [] if actions_info is None else actions_info[0]
        self.actions_e = [] if actions_info is None else actions_info[1]

        self.action_values = defaultdict(set)  # Ex: {north: [3.01, 2.00, 5.01]}

        self.maxlen_obs = 150
        self.maxlen_look = 150
        self.maxlen_inv = 50
        self.maxlen_action = 12
        self.simulation_num = getattr(args, "simulation_num", 1)
        self.use_llm = use_llm
        self.llm_policy = policy
        if use_llm and self.llm_policy is None:
            try:
                from mcts.virtualhome.llm_policy import (
                    LLMPolicy,
                )  # lazy import for backward compatibility

                self.llm_policy = LLMPolicy(device="cuda:0", model=getattr(args, "model", None))
            except Exception:
                self.llm_policy = None
        self.q_network = None
        self.state_dict = {}
        self.action_embedding = {}
        self.replay_file = replay_file
        self.pw_c = getattr(args, "pw_c", 2.0)
        self.pw_alpha = getattr(args, "pw_alpha", 0.5)
        self.pw_min_children = max(1, int(getattr(args, "pw_min_children", 6)))
        self.llm_prior_mode = getattr(args, "use_llm_prior", "all")
        if self.llm_prior_mode not in ("all", "root", "none"):
            self.llm_prior_mode = "all"
        self.prior_logs = []

    # ----------------------------
    # Utility
    # ----------------------------
    @staticmethod
    def state_id(history: list):
        return " ".join(history)

    def _use_llm_for_depth(self, depth: int) -> bool:
        if not self.use_llm:
            return False
        if self.llm_prior_mode == "none":
            return False
        if self.llm_prior_mode == "root":
            return depth == 0
        return True

    @staticmethod
    def _sorted_indices_from_probs(probs_full, length: int):
        try:
            arr = np.array(probs_full, dtype=np.float32)
            if len(arr) != length:
                return list(range(length))
            return list(np.argsort(-arr))
        except Exception:
            return list(range(length))

    def _progressive_widen_limit(self, state_node):
        if state_node is None or not state_node.valid_actions:
            return 0
        allowed = int(self.pw_c * (state_node.N ** self.pw_alpha)) if state_node.N > 0 else 0
        return min(len(state_node.valid_actions), max(self.pw_min_children, allowed))

    def _progressive_widen(self, state_node):
        """Expand children gradually based on visit count to cap branching factor."""
        if state_node is None:
            return
        if not state_node.valid_actions:
            state_node.children = []
            state_node.children_probs = []
            return
        limit = self._progressive_widen_limit(state_node)
        while len(state_node.children) < limit and state_node.remaining_action_indices:
            act_idx = state_node.remaining_action_indices.pop(0)
            action = state_node.valid_actions[act_idx]
            state_node.children.append(ActionNode(action))
            prob_pool = state_node.children_prob_pool or []
            if prob_pool and act_idx < len(prob_pool):
                prob = prob_pool[act_idx]
            else:
                prob = 1.0 / len(state_node.valid_actions)
            state_node.children_probs.append(prob)

    # ----------------------------
    # Policy scoring
    # ----------------------------
    def _score_actions_with_policy(
        self,
        history,
        ob,
        valid_actions,
        slot=None,
        payloads=None,
        state_signature=None,
    ):
        if valid_actions is None or len(valid_actions) == 0:
            return np.array([]), 0
        goal_text = ""
        goal_obj = getattr(self.env, "goal", None)
        if hasattr(self.env, "get_goal"):
            goal_text = self.env.get_goal()
        if self.llm_policy is not None:
            if hasattr(self.llm_policy, "score_prior"):
                try:
                    priors, pred_reward = self.llm_policy.score_prior(
                        obs=ob,
                        slot=slot,
                        actions=valid_actions,
                        payloads=payloads,
                        history=history,
                        goal=goal_obj or goal_text,
                        state_signature=state_signature,
                    )
                    if priors is not None and len(priors) == len(valid_actions):
                        self._log_prior(slot, valid_actions, priors, source="score_prior")
                        return priors, pred_reward
                except Exception:
                    pass
            if hasattr(self.llm_policy, "_calculate_emperical_prob"):
                return self.llm_policy._calculate_emperical_prob(
                    history, ob, valid_actions, goal_text, 10, self.round, self.discount_factor
                )
            if hasattr(self.llm_policy, "score_actions"):
                return self.llm_policy.score_actions(
                    history, ob, valid_actions, goal_text, self.round, self.discount_factor
                )
        # fallback to uniform
        uniform = np.ones((len(valid_actions),)) / len(valid_actions)
        return uniform, 0

    # ----------------------------
    # Build / Rebuild State
    # ----------------------------
    def rebuild_state(
        self,
        state,
        ob,
        history,
        valid_actions,
        done,
        reward=0,
        prev_action="<s>",
        use_llm=False,
    ):
        state.id = self.state_id(history)
        state.valid_actions = valid_actions if valid_actions is not None else []
        state.use_llm = use_llm
        slot = getattr(self.env, "last_slot", None)
        payloads = getattr(self.env, "action_payloads", None)
        state_signature = None
        try:
            env_state = getattr(self.env, "state", None)
            if env_state is not None and hasattr(env_state, "signature"):
                state_signature = env_state.signature()
        except Exception:
            state_signature = None

        # children_probs
        if not use_llm:
            if not state.valid_actions:
                probs_full = np.array([])
            else:
                probs_full = np.ones((len(state.valid_actions),)) / len(state.valid_actions)
        else:
            probs_full, state.predicted_reward = self._score_actions_with_policy(
                history,
                ob,
                state.valid_actions,
                slot=slot,
                payloads=payloads,
                state_signature=state_signature,
            )

        self.state_dict[state.id] = state
        state.children = []
        state.children_prob_pool = probs_full.tolist() if hasattr(probs_full, "tolist") else list(probs_full)
        state.children_probs = []
        state.remaining_action_indices = self._sorted_indices_from_probs(
            state.children_prob_pool, len(state.valid_actions)
        )
        self._progressive_widen(state)

        return state

    # ----------------------------
    # Logging helpers
    # ----------------------------
    def _format_slot(self, slot):
        if slot is None:
            return "slot=None"
        parts = [f"type={getattr(slot, 'type', None)}"]
        for key in ("day", "meal_type", "seg", "city", "origin", "destination"):
            val = getattr(slot, key, None)
            if val is not None:
                parts.append(f"{key}={val}")
        return " ".join(parts)

    def _log_prior(self, slot, actions, priors, source=""):
        entry = {}
        try:
            slot_txt = self._format_slot(slot)
            arr = np.array(priors, dtype=np.float32)
            order = list(np.argsort(-arr))
            topk = order[: min(3, len(order))]
            top_list = []
            for idx in topk:
                prob = float(arr[idx])
                act = actions[idx] if idx < len(actions) else "<out_of_range>"
                top_list.append({"index": int(idx), "p": prob, "action": act})
            entry = {
                "slot": slot_txt,
                "source": source or "llm",
                "candidates": len(actions),
                "top": top_list,
            }
            self.prior_logs.append(entry)
            if self.debug:
                header = f"[PRIOR] {source or 'llm'} {slot_txt}"
                print(header)
                for item in top_list:
                    print(f"  [{item['index']}] p={item['p']:.3f} | {item['action']}")
        except Exception:
            try:
                self.prior_logs.append(entry or {"slot": "error", "source": source or "llm"})
            except Exception:
                pass

    def build_state(
        self,
        ob,
        history,
        valid_actions,
        done,
        reward=0,
        prev_action="<s>",
        use_llm=False,
    ):
        state = StateNode()
        state.ob = ob
        state.state = ob
        state.done = done
        state.reward = reward
        state.prev_action = prev_action
        state.history = history
        state.id = self.state_id(history)
        state.valid_actions = valid_actions if valid_actions is not None else []
        state.use_llm = use_llm
        slot = getattr(self.env, "last_slot", None)
        payloads = getattr(self.env, "action_payloads", None)
        state_signature = None
        try:
            env_state = getattr(self.env, "state", None)
            if env_state is not None and hasattr(env_state, "signature"):
                state_signature = env_state.signature()
        except Exception:
            state_signature = None

        # children_probs
        if not use_llm:
            if not state.valid_actions:
                probs_full = np.array([])
            else:
                probs_full = np.ones((len(state.valid_actions),)) / len(state.valid_actions)
        else:
            probs_full, state.predicted_reward = self._score_actions_with_policy(
                history,
                ob,
                state.valid_actions,
                slot=slot,
                payloads=payloads,
                state_signature=state_signature,
            )

        self.state_dict[state.id] = state
        state.children = []
        state.children_prob_pool = probs_full.tolist() if hasattr(probs_full, "tolist") else list(probs_full)
        state.children_probs = []
        state.remaining_action_indices = self._sorted_indices_from_probs(
            state.children_prob_pool, len(state.valid_actions)
        )
        self._progressive_widen(state)

        return state

    # ----------------------------
    # Core MCTS: search / simulate / rollout
    # ----------------------------
    def search(self, ob, history, cur_depth, valid_actions, done):
        """
        Search the best action with probs
        :return: best action (or None if truly no action)
        """
        init_history = history.copy()

        # 建立根节点
        self.root = self.build_state(ob, history, valid_actions, done, use_llm=self._use_llm_for_depth(0))

        # 如果一开始就无动作，直接返回 None
        if not self.root.valid_actions:
            if self.debug:
                print("[DEBUG] search: root has no valid_actions, returning None")
            return None

        # 多次仿真
        for _ in range(self.simulation_num):
            # 重建真实状态，而不是仅设置 history
            self.env.replay(init_history)
            _, root = self.simulate(self.root, 0)
            self.root = root

        # 选择 root 的最佳动作
        best_action_node_idx = self.greedy_action_node(self.root, 0, 0, if_print=self.debug)

        if best_action_node_idx is None:
            # 没有可选的 child，fallback：如果 root 还有 valid_actions，就直接返回第一个
            if self.debug:
                print("[DEBUG] search: greedy_action_node returned None at root")
            if self.root.valid_actions:
                return self.root.valid_actions[0]
            # 连 valid_actions 都没有 → 彻底无动作
            return None

        best_action_node = self.root.children[best_action_node_idx]
        self.root.best_action_node = best_action_node
        return self.root.best_action_node.action

    def simulate(self, state_node, depth):
        # 终止条件
        if state_node.done or depth == self.max_depth:
            return 0, state_node

        self._progressive_widen(state_node)

        # 无 children → 死路（没有后续动作）
        if not state_node.children or len(state_node.children) == 0:
            state_node.done = True
            return 0, state_node

        best_action_node_idx = self.greedy_action_node(
            state_node, self.exploration_constant, self.bonus_constant
        )
        if best_action_node_idx is None:
            # 当前节点下没有可扩展的动作，标记为终止
            state_node.done = True
            return 0, state_node

        best_action_node = state_node.children[best_action_node_idx]
        rollout_next = False

        ob, reward, done, history, valid_actions = self.env.step(best_action_node.action)
        next_state_id = self.state_id(history)

        if next_state_id == best_action_node.children_id:
            next_state_node = best_action_node.children
            if next_state_node.use_llm is False:
                next_state_node = self.build_state(
                    ob,
                    history,
                    valid_actions,
                    done,
                    reward,
                    prev_action=best_action_node.action,
                    use_llm=self._use_llm_for_depth(depth + 1),
                )
                next_state_node.parent = state_node
                rollout_next = True
        else:
            next_state_node = self.build_state(
                ob,
                history,
                valid_actions,
                done,
                reward,
                prev_action=best_action_node.action,
                use_llm=self._use_llm_for_depth(depth + 1),
            )
            next_state_node.parent = state_node
            best_action_node.children = next_state_node
            best_action_node.children_id = next_state_node.id
            rollout_next = True

        if rollout_next:
            rollout_r = []
            for _ in range(1):
                random_r = reward + self.discount_factor * self.rollout(next_state_node, depth + 1)
                rollout_r.append(random_r)
            R = sum(rollout_r) / len(rollout_r)
        else:
            r, next_state_node = self.simulate(next_state_node, depth + 1)
            R = reward + self.discount_factor * r

        state_node.N += 1
        best_action_node.N += 1
        best_action_node.children = next_state_node
        best_action_node.Rs.append(R)
        if len(best_action_node.Rs) > 0:
            best_action_node.Q = np.sum(
                np.array(best_action_node.Rs) * utils.softmax(best_action_node.Rs, T=10)
            )
        state_node.best_action_node = best_action_node
        return R, state_node

    def rollout(self, state_node, depth):
        if state_node.done or depth == self.max_depth:
            return 0.0
        self._progressive_widen(state_node)
        if not state_node.children or len(state_node.children) == 0:
            return 0.0

        action_node = np.random.choice(state_node.children, 1)[0]
        action = action_node.action

        ob, reward, done, history, valid_actions = self.env.step(action)
        next_state_id = self.state_id(history)

        if next_state_id == action_node.children_id:
            next_state_node = action_node.children
        else:
            next_state_node = self.build_state(
                ob,
                history,
                valid_actions,
                done,
                reward,
                prev_action=action,
                use_llm=self._use_llm_for_depth(depth + 1),
            )
            next_state_node.parent = state_node
            action_node.children = next_state_node
            action_node.children_id = next_state_node.id

        r = reward + self.discount_factor * self.rollout(next_state_node, depth + 1)
        return r

    # ----------------------------
    # Action selection helpers
    # ----------------------------
    def max_visit_action_node(self, state_node):
        children_count = np.array([c.N for c in state_node.children], dtype=np.float32)
        if np.max(children_count) == 0:
            # 全 0：退化成均匀采样
            return np.random.choice(state_node.children, 1)[0]
        children_count = children_count / np.max(children_count)
        count_based_probs = children_count ** (1 / self.action_selection_temp) / (
            np.sum(children_count ** (1 / self.action_selection_temp))
        )
        return np.random.choice(state_node.children, p=count_based_probs)

    def greedy_action_node(self, state_node, exploration_constant, bonus_constant, if_print=False):
        # 这里是关键补丁：children 为空时直接返回 None，而不是崩溃
        if not state_node.children or len(state_node.children) == 0:
            if self.debug:
                print(
                    f"[DEBUG] greedy_action_node: empty children for state id={state_node.id} "
                    f"history={state_node.history}"
                )
            return None

        best_value = -np.inf
        best_children = []
        best_children_prob = []

        for i in range(len(state_node.children)):
            child = state_node.children[i]
            if len(state_node.children_probs) != len(state_node.children):
                # 不一致时，回退为均匀概率
                state_node.children_probs = np.ones((len(state_node.children),)) / len(state_node.children)
            child_prob = state_node.children_probs[i]

            if exploration_constant == 0:
                ucb_value = child.Q
            elif self.uct_type == "UCT":
                ucb_value = child.Q + exploration_constant * np.sqrt(
                    np.log(state_node.N + 1) / (child.N + 1)
                )
            elif self.uct_type == "PUCT":
                ucb_value = child.Q + exploration_constant * child_prob * np.sqrt(state_node.N) / (
                    child.N + 1
                )
            elif self.uct_type == "MC-LAVE":
                if child.action in self.action_embedding.keys():
                    action_e = self.action_embedding[child.action]
                else:
                    action_e = utils.vectorize(child.action)
                    self.action_embedding[child.action] = action_e

                actions = list(self.action_values.keys())
                if child.action in actions:
                    actions.pop(actions.index(child.action))

                actions_e = []
                for a in actions:
                    actions_e.append(self.action_embedding[a])

                near_act, near_idx = utils.find_near_actions(
                    action_e, actions, np.array(actions_e), threshold=0.8
                )
                if len(near_idx) == 0:
                    child.Q_hat = 0
                else:
                    near_Qs = set()
                    for a in near_act:
                        near_Qs.add(np.mean(list(self.action_values[a])))
                    near_Qs = list(near_Qs)
                    child.Q_hat = utils.softmax_value(near_Qs)

                ucb_value = (
                    child.Q
                    + exploration_constant
                    * np.sqrt(state_node.N + 1)
                    / (child.N + 1)
                    * child_prob
                    + bonus_constant
                    * np.sqrt(state_node.N + 1)
                    / (child.N + 1)
                    * child.Q_hat
                )
            else:
                raise NotImplementedError

            if ucb_value == best_value:
                best_children.append(i)
                best_children_prob.append(child_prob)
            elif ucb_value > best_value:
                best_value = ucb_value
                best_children = [i]
                best_children_prob = [child_prob]

        if if_print:
            for c in state_node.children:
                if c.N > 0:
                    print(c.action, c.Q, c.N)

        best_children_prob = np.array(best_children_prob, dtype=np.float32)
        prob_sum = np.sum(best_children_prob)
        if prob_sum <= 0:
            # 所有概率都是 0 或数值问题 → 随机从 best_children 里挑一个
            return best_children[0]

        best_children_prob = best_children_prob / prob_sum
        output_action_index = np.argmax(best_children_prob)
        return best_children[output_action_index]

    # ----------------------------
    # Debug helper
    # ----------------------------
    def root_statistics(self, top_k: int = 5):
        """Return sorted root child stats for debugging."""
        if self.root is None or not self.root.children:
            return []
        stats = []
        for child in self.root.children:
            stats.append(
                {
                    "action": child.action,
                    "Q": child.Q,
                    "N": child.N,
                }
            )
        stats = sorted(stats, key=lambda x: (x["Q"], x["N"]), reverse=True)
        return stats[:top_k]
