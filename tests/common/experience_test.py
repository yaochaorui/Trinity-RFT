# -*- coding: utf-8 -*-
"""Test cases for Storage modules."""
import os
import unittest

import torch

from trinity.buffer.schema.sql_schema import ExperienceModel
from trinity.common.experience import EID, Experience, Experiences

db_url = os.path.join(os.path.dirname(__file__), "tmp", "test.db")
dataset_path = os.path.join(os.path.dirname(__file__), "data")


class TestEID(unittest.TestCase):
    def test_eid_properties(self):
        # test properties
        eid = EID(batch=1, task=2, run=3, step=4, suffix="abc123")
        self.assertEqual(eid.uid, "1/2/3/4/abc123")
        self.assertEqual(eid.sid, "1/2/4")
        self.assertEqual(eid.rid, "1/2/3")
        self.assertEqual(eid.tid, "1/2")
        self.assertEqual(str(eid), "1/2/3/4/abc123")
        self.assertIn("EID(batch=1, task=2, run=3, step=4, uuid=abc123)", repr(eid))

        # test unique
        eid1 = EID(batch=1, task=2, run=3, step=4)
        eid2 = EID(batch=1, task=2, run=3, step=4)
        self.assertNotEqual(eid1.suffix, eid2.suffix)
        self.assertNotEqual(eid1.uid, eid2.uid)

        # test default
        eid = EID()
        eid2 = EID()
        self.assertIsInstance(eid.suffix, str)
        self.assertEqual(eid.batch, 0)
        self.assertEqual(eid.task, 0)
        self.assertEqual(eid.run, 0)
        self.assertEqual(eid.step, 0)
        self.assertNotEqual(eid.uid, eid2.uid)


class TestExperience(unittest.TestCase):
    def test_single_turn_experience(self):
        tokens = torch.tensor([10, 11, 12], dtype=torch.int32)
        logprobs = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        exp = Experience(tokens=tokens, logprobs=logprobs, reward=1.0, prompt_length=1)
        self.assertEqual(exp.experience_type.name, "SINGLE_TURN")
        self.assertTrue(torch.equal(exp.tokens, tokens))
        self.assertTrue(torch.equal(exp.logprobs, logprobs))
        self.assertEqual(exp.reward, 1.0)
        self.assertEqual(exp.prompt_length, 1)
        self.assertTrue(torch.equal(exp.action_mask, torch.tensor([0, 1, 1], dtype=torch.bool)))

    def test_multi_turn_experience(self):
        tokens = torch.tensor([1, 2, 3, 4])
        logprobs = torch.tensor([0.1, 0.2, 0.3, 0.4])
        action_mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
        exp = Experience(tokens=tokens, logprobs=logprobs, reward=2.0, action_mask=action_mask)
        self.assertEqual(exp.experience_type.name, "MULTI_TURN")
        self.assertTrue(torch.equal(exp.action_mask, action_mask))
        self.assertEqual(exp.prompt_length, 1)

    def test_dpo_experience(self):
        tokens = torch.tensor([1, 2])
        chosen = torch.tensor([3, 4])
        rejected = torch.tensor([5, 6])
        exp = Experience(tokens=tokens, chosen=chosen, rejected=rejected, reward=0.5)
        self.assertEqual(exp.experience_type.name, "DPO")
        self.assertTrue(torch.equal(exp.chosen, chosen))
        self.assertTrue(torch.equal(exp.rejected, rejected))
        self.assertEqual(exp.prompt_length, 2)

    def test_serialize_deserialize(self):
        tokens = torch.tensor([1, 2, 3])
        exp = Experience(tokens=tokens, reward=1.23, prompt_length=1)
        data = exp.serialize()
        exp2 = Experience.deserialize(data)
        self.assertTrue(torch.equal(exp.tokens, exp2.tokens))
        self.assertEqual(exp.reward, exp2.reward)
        self.assertEqual(exp.prompt_length, exp2.prompt_length)
        self.assertEqual(exp.experience_type, exp2.experience_type)

    def test_to_dict(self):
        tokens = torch.tensor([1, 2, 3])
        exp = Experience(
            tokens=tokens, reward=2.5, prompt_length=1, prompt_text="hi", response_text="yo"
        )
        d = exp.to_dict()
        self.assertIn("eid", d)
        self.assertIn("type", d)
        self.assertIn("reward", d)
        self.assertEqual(d["prompt_text"], "hi")
        self.assertEqual(d["response_text"], "yo")
        self.assertEqual(d["reward"], 2.5)

    def test_gather(self):
        # test empty gathering
        batch = Experiences.gather_experiences([])
        self.assertEqual(batch.tokens.numel(), 0)
        self.assertEqual(batch.rewards.numel(), 0)
        self.assertEqual(batch.eids, [])

        # test single experience gathering
        exp = Experience(tokens=torch.tensor([1, 2, 3]), reward=1.0, prompt_length=1)
        batch = Experiences.gather_experiences([exp])
        self.assertEqual(batch.batch_size, 1)
        self.assertTrue(
            torch.equal(batch.tokens[0], torch.tensor([0, 1, 2, 3], dtype=torch.int64)[-3:])
        )
        self.assertEqual(batch.prompt_length, 1)
        self.assertEqual(batch.rewards[0], 1.0)

        # test multiple experiences gathering
        exps = [
            Experience(tokens=torch.tensor([1, 2]), reward=0.1, prompt_length=1),
            Experience(tokens=torch.tensor([3, 4, 5]), reward=0.2, prompt_length=2),
        ]
        batch = Experiences.gather_experiences(exps)
        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.prompt_length, 2)
        self.assertEqual(batch.tokens.shape[1], 3)
        self.assertEqual(batch.rewards[0], 0.1)
        self.assertEqual(batch.rewards[1], 0.2)

    def test_action_mask_and_logprobs_type(self):
        exp = Experience(tokens=[1, 2, 3], logprobs=[0.1, 0.2, 0.3], prompt_length=1)
        self.assertIsInstance(exp.tokens, torch.Tensor)
        self.assertIsInstance(exp.logprobs, torch.Tensor)
        self.assertIsInstance(exp.action_mask, torch.Tensor)

    def test_assertions(self):
        # prompt_length must be > 0
        with self.assertRaises(AssertionError):
            Experience(tokens=[1, 2, 3], prompt_length=0)
        # tokens must be longer than prompt_length for single-turn
        with self.assertRaises(AssertionError):
            Experience(tokens=[1, 2], prompt_length=2)
        # DPO: tokens must match prompt_length
        exp = Experience(tokens=[1, 2], chosen=[3], rejected=[4], prompt_length=1)
        exp.prompt_length = 2  # should automatically adjust


class TestExperienceConversion(unittest.TestCase):
    """Test cases for ExperienceModel"""

    def test_experience_model_experience_conversion(self):
        """Test the conversion between Experience and ExperienceModel"""
        tokens = torch.tensor([1, 2, 3], dtype=torch.int32)
        reward = 0.6
        prompt_length = 2
        logprobs = torch.tensor([0, 0, 0.1], dtype=torch.float32)
        experience = Experience(
            tokens=tokens,
            reward=reward,
            prompt_length=prompt_length,
            logprobs=logprobs,
        )

        model = ExperienceModel.from_experience(experience)
        new_experience = model.to_experience()
        self.assertTrue(torch.equal(new_experience.tokens, tokens))
        self.assertEqual(new_experience.prompt_length, prompt_length)
        self.assertEqual(new_experience.reward, reward)
        self.assertTrue(torch.equal(new_experience.logprobs, logprobs))
        self.assertTrue(torch.equal(new_experience.action_mask, experience.action_mask))

    def test_batch_conversion(self):
        exps = [
            Experience(
                tokens=torch.tensor([1, 2]),
                prompt_length=1,
                reward=float(0.1),
                logprobs=torch.tensor([0, 0.1]),
            ),
            Experience(
                tokens=torch.tensor([1, 2, 3]),
                prompt_length=2,
                reward=float(0.2),
                logprobs=torch.tensor([0, 0, 0.1]),
            ),
        ]
        batch = Experiences.gather_experiences(exps)
        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.prompt_length, 2)
        prompt_length = batch.prompt_length
        for i in range(batch.batch_size):
            self.assertEqual(batch.rewards[i], exps[i].reward)
            self.assertTrue(
                torch.all(
                    batch.tokens[i][
                        prompt_length
                        - exps[i].prompt_length : prompt_length
                        - exps[i].prompt_length
                        + exps[i].tokens.size(0)
                    ]
                    == exps[i].tokens
                )
            )
            self.assertTrue(
                torch.all(
                    batch.logprobs[i][
                        prompt_length
                        - exps[i].prompt_length : prompt_length
                        + exps[i].tokens.size(0)
                        - exps[i].prompt_length
                    ]
                    == exps[i].logprobs
                )
            )
            self.assertTrue(
                torch.all(
                    batch.action_masks[i][
                        prompt_length
                        - exps[i].prompt_length : prompt_length
                        - exps[i].prompt_length
                        + exps[i].action_mask.size(0)
                    ]
                    == exps[i].action_mask
                )
            )

    def test_multiturn_experience_batch_converstion(self):
        exps = [
            Experience(
                tokens=torch.tensor([1, 2, 3, 4]),
                reward=float(0.3),
                logprobs=torch.tensor([0, 0, 0.1, 0.2]),
                action_mask=torch.tensor([1, 0, 1, 0]),
            ),
            Experience(
                tokens=torch.tensor([1, 2, 3, 4]),
                reward=float(0.4),
                logprobs=torch.tensor([0, 0, 0, 0.1]),
                action_mask=torch.tensor([1, 0, 0, 1]),
            ),
        ]
        batch = Experiences.gather_experiences(exps)
        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.prompt_length, 1)
        prompt_length = batch.prompt_length
        for i in range(batch.batch_size):
            self.assertEqual(batch.rewards[i], exps[i].reward)
            self.assertTrue(
                torch.all(
                    batch.tokens[i][
                        prompt_length
                        - exps[i].prompt_length : prompt_length
                        - exps[i].prompt_length
                        + exps[i].tokens.size(0)
                    ]
                    == exps[i].tokens
                )
            )
            self.assertTrue(
                torch.all(
                    batch.logprobs[i][
                        prompt_length
                        - exps[i].prompt_length : prompt_length
                        + exps[i].tokens.size(0)
                        - exps[i].prompt_length
                    ]
                    == exps[i].logprobs
                )
            )
            self.assertTrue(
                torch.all(
                    batch.action_masks[i][
                        prompt_length
                        - exps[i].prompt_length : prompt_length
                        - exps[i].prompt_length
                        + exps[i].action_mask.size(0)
                    ]
                    == exps[i].action_mask
                )
            )

    def test_dpo_experience_batch_conversion(self):
        exps = [
            Experience(
                tokens=torch.tensor([1, 2]),
                chosen=torch.tensor([3, 4]),
                rejected=torch.tensor([5, 6]),
            ),
            Experience(
                tokens=torch.tensor([7, 8, 9]),
                chosen=torch.tensor([10, 11]),
                rejected=torch.tensor([12, 13]),
            ),
        ]
        batch = Experiences.gather_experiences(exps)
        self.assertEqual(batch.batch_size, 4)
        self.assertEqual(batch.prompt_length, 3)
        prompt_length = batch.prompt_length
        for i in range(batch.batch_size):
            j = i // 2
            self.assertTrue(
                torch.all(
                    batch.tokens[i][
                        prompt_length
                        - exps[j].prompt_length : prompt_length
                        - exps[j].prompt_length
                        + exps[j].tokens.size(0)
                    ]
                    == exps[j].tokens
                )
            )


if __name__ == "__main__":
    unittest.main()
