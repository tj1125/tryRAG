"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2507221338
"""


from dataclasses import dataclass
import numpy as np

@dataclass
class DocChunk:
    idx: int
    url: str
    content: str
    embedding: np.ndarray = None
    score = None

@dataclass
class lookupQuery:
    question: str












