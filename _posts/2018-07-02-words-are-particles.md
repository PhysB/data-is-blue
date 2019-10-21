---
title:  "Words are particles!"
date:   2018-07-02 12:00:00 -0700
categories: nlp
# excerpt: "Part 1 of two-part post on word2vec and natural language processing."
---
I have a new project at work that means that I get to dive into a new field - natural language processing. My boss and coworkers suggested a few papers for me to read to lay a foundation, and I'm going to write up each one as I go through it. This is, frankly, way more helpful for me than to anyone who might be reading this. However, if you were ever curious about how a physicist interprets and understands natural language processing, well, here's your look inside my brain.

I'm starting with one of the absolute classic papers in the field: ["Distributed Representations of Words and Phrases and their Compositionality."](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) This is the original Google word2vec paper that blew everyone's minds with the ability to add and subtract words in a meaningful way. E.g. "king-male+female = queen." Let's dive in.

<!--more-->
I suppose I should add all the caveats - this isn't really intended to be a tutorial and I'll probably get things wrong. I'm just trying to write things up as I understand them - putting the  Feynman method to work here.

We start with the Skip-gram model, which says that given a word in a sentence, we want to predict which words are most likely to be nearby. So, given the word cloud, we might expect the word rain to occur more frequently in that sentence, than, say potato.

Formally, we want to maximize the equation: 

$$\frac{1}{T} \sum_{t=1}^T \sum_{-c <= j < c, j \neq 0} \log{p(w_{t+j} \vert w_t)}$$

This says that, given the word $w_t$, we want most probable $j$ words surrounding that word. We learn that by going through all the training words (there are $T$of them), and looking at the $\pm c$ words around the word in question.

Ok, given that this is the objective, the first thing we need to do is define that inner probability function, $p(w_{t+j} \vert w_t)$. The Skip-gram model does that with the softmax function.

$$p(w_O \vert w_I) = \frac{e^{ \left( {v'}_{w_O}^\top v_{w_I} \right) }}{\sum_{w=1}^W e^{\left( {v'}_w^\top v_{w_I} \right) }}$$

Alright. Now I've got a lot to parse here. The variables $v_w$ and $v'_w$ are the input and output vector representations of a word $w$ and there are $W$ words in the vocabulary.

Hmmmm. At first blush, this seems like it makes sense. We want the probability of some output word, given some input word. If I understand correctly, this equation should tell me that p(rain \| cloud) is bigger than p(potato \| cloud). The denominator of this equation makes sense to me--it's a normalization over all the words in the vocabulary. But what does it mean to take the dot product of the output representation of the output word with input representation of the input word (the thing in the exponential in the numerator)?

I'm struggling to make sense of this, because the difference between the input and output representations isn't totally clear to me. Then it occurs to me, this looks remarkably similar to the Boltzmann distribution!

$$ p_i = \frac{e^{ \left(-\varepsilon_i / kT \right)}}{\sum_{j=1}^M e^{\left(  -\varepsilon_j / kT \right)}} $$

This equation gives the probability for a particle in a system to be in state $i$, with energy $\varepsilon_i $, where k is the Boltzmann constant and T is the temperature of the system. The denominator is the canonical partition function (Z), so this can be rewritten as

$$p_i = \frac{1}{Z} e^{-\varepsilon_i / kT} $$

When we take the ratio of the probabilities of two particles in two different states, we get the following:

$$ \frac{p_i}{p_j} = e^{(\varepsilon_j - \varepsilon_i) / kT} $$

This says that particles in low energy states are more probable than particles in high energy states. What happens if we play the same trick with words? Well, we get something like this:

$$ \frac{p(w_j \vert w_I)}{p(w_k \vert w_I)}  = \frac{\exp{\left({v'}_{w_j}^\top v_{w_I} \right)}}{\exp{\left({v'}_{w_k}^\top v_{w_I} \right)}} = \exp\left({v'}_{w_j}^\top v_{w_I} -{v'}_{w_k}^\top v_{w_I}\right)$$

Given the same input word, an output word is more probable if its inner product with that input word is higher.

That is, if $$v'_{w_j} \cdot v_{w_I}$$ is larger than $$v'_{w_k} \cdot v_{w_I}$$ then word j is more probable than word k.

Thinking more about these variables, $v'_{w_j}$ is the output representation of the output word. If it is orthogonal to the input word, this dot product is going to be near zero. If it is aligned (or nearly so) with the input word, the dot product is going to be near one, which maximizes the probability equation.

So, given the input word and our two choices for the output word (j and k), the most probable output word is the one that more aligned with it in the vector space. That makes sense to me. Location, location, location. It's still not totally clear to me (yet) what the difference is between the input and output representation, but I think I'm starting to get a handle on this.

The Boltzmann distribution maximizes the entropy of our system of particles:

$$H = - \sum_{i=1}^M p_i \log p_i $$

There are lots of interesting debates about the equivalence of thermodynamic and information entropy, which I am going to completely neglect here. Consider all of the following as merely an extended analogy.

We maximize entropy by having lots of high probability, low energy particles. We minimize entropy by having lots of low probability, high energy particles. Minimizing entropy is like maximizing information. (Minimum disorder = high information). So words/particles that minimize entropy and maximize information are high energy and low probability.

I imagine all my low energy particles/words as the highly common ones: the, to, an, and. Not much information in those. That means that my high energy particles/words are the improbable ones: defenestration, widdershins, ragamuffin. Lots of information in those words! Of course, this breaks down when you consider the extremes. This would mean that extremely obscure words also contain the most information, which, maybe? By some definition of information?

Anyway, moving back to the original equation, we have to normalize by all the words/particles in the vocabulary/system in order to calculate the actual probability $p(w_O \vert w_I) $. That's the summation in the denominator.

It turns out that calculating that sum is computationally expensive, which is what the rest of the paper deals with. As this is getting a bit long, I'll deal with that in the second part of this write-up. Spoiler alert: it's approximations all the way down.
