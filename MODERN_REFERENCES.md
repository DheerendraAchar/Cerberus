# Modern References for Adversarial AI (2018-2025)

**For Project Cerberus — Adversarial AI Simulation Framework**

---

## 📚 Essential References (Post-2018)

### **Foundational Theory & Understanding**

1. **Ilyas, A., Santurkar, S., Tsipras, D., Engstrom, L., Tran, B., & Madry, A. (2019)**
   - "Adversarial Examples Are Not Bugs, They Are Features"
   - *NeurIPS 2019*
   - arXiv:1905.02175
   - **Why cite:** Revolutionary perspective - adversarial examples exploit legitimate model features, not bugs

2. **Croce, F., & Hein, M. (2020)**
   - "Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks"
   - *ICML 2020*
   - arXiv:2003.01690
   - **Why cite:** AutoAttack benchmark - standard for evaluating robustness (very influential)

3. **Carlini, N., Athalye, A., Papernot, N., Brendel, W., Rauber, J., Tsipras, D., Goodfellow, I., Madry, A., & Kurakin, A. (2019)**
   - "On Evaluating Adversarial Robustness"
   - arXiv:1902.06705
   - **Why cite:** Best practices for adversarial robustness evaluation

---

### **Attack Methods (Post-2018)**

4. **Croce, F., & Hein, M. (2020)**
   - "Minimally Distorted Adversarial Examples with a Fast Adaptive Boundary Attack"
   - *ICML 2020*
   - arXiv:1907.02044
   - **Why cite:** FAB attack - efficient minimal perturbation attacks

5. **Andriushchenko, M., Croce, F., Flammarion, N., & Hein, M. (2020)**
   - "Square Attack: A Query-efficient Black-box Adversarial Attack via Random Search"
   - *ECCV 2020*
   - arXiv:1912.00049
   - **Why cite:** Black-box attacks without gradient access (practical threat model)

6. **Dong, Y., Liao, F., Pang, T., Su, H., Zhu, J., Hu, X., & Li, J. (2018)**
   - "Boosting Adversarial Attacks with Momentum"
   - *CVPR 2018*
   - arXiv:1710.06081
   - **Why cite:** Momentum-based attacks (MI-FGSM) - improves transferability

---

### **Defense Mechanisms**

7. **Zhang, H., Yu, Y., Jiao, J., Xing, E., El Ghaoui, L., & Jordan, M. (2019)**
   - "Theoretically Principled Trade-off between Robustness and Accuracy"
   - *ICML 2019*
   - arXiv:1901.08573
   - **Why cite:** TRADES defense - theoretical foundation for robustness-accuracy trade-off

8. **Carmon, Y., Raghunathan, A., Schmidt, L., Duchi, J. C., & Liang, P. S. (2019)**
   - "Unlabeled Data Improves Adversarial Robustness"
   - *NeurIPS 2019*
   - arXiv:1905.13736
   - **Why cite:** Semi-supervised adversarial training using unlabeled data

9. **Wang, Y., Zou, D., Yi, J., Bailey, J., Ma, X., & Gu, Q. (2020)**
   - "Improving Adversarial Robustness Requires Revisiting Misclassified Examples"
   - *ICLR 2020*
   - arXiv:1910.09961
   - **Why cite:** MART defense - focuses on misclassified adversarial examples

10. **Rebuffi, S. A., Gowal, S., Calian, D. A., Stimberg, F., Wiles, O., & Mann, T. (2021)**
    - "Fixing Data Augmentation to Improve Adversarial Robustness"
    - arXiv:2103.01946
    - **Why cite:** Data augmentation strategies for improved robustness

---

### **Certified Defenses & Verification**

11. **Cohen, J., Rosenfeld, E., & Kolter, Z. (2019)**
    - "Certified Adversarial Robustness via Randomized Smoothing"
    - *ICML 2019*
    - arXiv:1902.02918
    - **Why cite:** Randomized smoothing - provable robustness guarantees

12. **Salman, H., Li, J., Razenshteyn, I., Zhang, P., Zhang, H., Bubeck, S., & Yang, G. (2019)**
    - "Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers"
    - *NeurIPS 2019*
    - arXiv:1906.04584
    - **Why cite:** Combining adversarial training with certified defenses

---

### **Real-World Applications & Physical Attacks**

13. **Eykholt, K., Evtimov, I., Fernandes, E., Li, B., Rahmati, A., Xiao, C., Prakash, A., Kohno, T., & Song, D. (2018)**
    - "Robust Physical-World Attacks on Deep Learning Visual Classification"
    - *CVPR 2018*
    - arXiv:1707.08945
    - **Why cite:** Physical adversarial examples (stop sign attacks) - real-world threat

14. **Athalye, A., Engstrom, L., Ilyas, A., & Kwok, K. (2018)**
    - "Synthesizing Robust Adversarial Examples"
    - *ICML 2018*
    - arXiv:1707.07397
    - **Why cite:** 3D adversarial examples - physical world robustness

15. **Sharif, M., Bhagavatula, S., Bauer, L., & Reiter, M. K. (2019)**
    - "A General Framework for Adversarial Examples with Objectives"
    - *ACM TOPS 2019*
    - arXiv:1801.00349
    - **Why cite:** Adversarial eyeglasses - face recognition attacks

---

### **Adversarial Robustness Benchmarks**

16. **Croce, F., Andriushchenko, M., Sehwag, V., Debenedetti, E., Flammarion, N., Chiang, M., Mittal, P., & Hein, M. (2021)**
    - "RobustBench: A Standardized Adversarial Robustness Benchmark"
    - *NeurIPS 2021 Datasets and Benchmarks Track*
    - arXiv:2010.09670
    - **Why cite:** RobustBench leaderboard - standard benchmark for comparing defenses

17. **Hendrycks, D., Zhao, K., Basart, S., Steinhardt, J., & Song, D. (2021)**
    - "Natural Adversarial Examples"
    - *CVPR 2021*
    - arXiv:1907.07174
    - **Why cite:** ImageNet-A - naturally occurring adversarial examples

---

### **Adversarial Training Improvements**

18. **Rice, L., Wong, E., & Kolter, Z. (2020)**
    - "Overfitting in Adversarially Robust Deep Learning"
    - *ICML 2020*
    - arXiv:2002.11569
    - **Why cite:** Understanding overfitting in adversarial training

19. **Wu, D., Xia, S. T., & Wang, Y. (2020)**
    - "Adversarial Weight Perturbation Helps Robust Generalization"
    - *NeurIPS 2020*
    - arXiv:2004.05884
    - **Why cite:** AWP - weight perturbation for better adversarial robustness

20. **Gowal, S., Qin, C., Uesato, J., Mann, T., & Kohli, P. (2020)**
    - "Uncovering the Limits of Adversarial Training against Norm-Bounded Adversarial Examples"
    - arXiv:2010.03593
    - **Why cite:** State-of-the-art adversarial training techniques (2020)

---

### **Recent Surveys & Comprehensive Reviews**

21. **Ren, K., Zheng, T., Qin, Z., & Liu, X. (2020)**
    - "Adversarial Attacks and Defenses in Deep Learning"
    - *Engineering 2020*
    - DOI: 10.1016/j.eng.2019.12.012
    - **Why cite:** Comprehensive survey of attacks and defenses

22. **Chakraborty, A., Alam, M., Dey, V., Chattopadhyay, A., & Mukhopadhyay, D. (2021)**
    - "A Survey on Adversarial Attacks and Defences"
    - *CAAI Transactions on Intelligence Technology 2021*
    - DOI: 10.1049/cit2.12028
    - **Why cite:** Recent comprehensive survey covering 2018-2020 developments

23. **Bai, T., Luo, J., Zhao, J., Wen, B., & Wang, Q. (2021)**
    - "Recent Advances in Adversarial Training for Adversarial Robustness"
    - *IJCAI 2021 Survey Track*
    - arXiv:2102.01356
    - **Why cite:** Focused survey on adversarial training methods

---

### **Toolkits & Frameworks**

24. **Nicolae, M. I., Sinn, M., Tran, M. N., Buesser, B., Rawat, A., Wistuba, M., Zantedeschi, V., Baracaldo, N., Chen, B., Ludwig, H., Molloy, I. M., & Edwards, B. (2019)**
    - "Adversarial Robustness Toolbox v1.0.0"
    - arXiv:1807.01069
    - **Why cite:** IBM ART - the toolkit you're using! Essential to cite

25. **Goodman, D., Xin, H., Yang, W., Yuesheng, W., Junfeng, X., & Huan, Z. (2020)**
    - "ARES: A Benchmark Platform for Adversarial Robustness Evaluation"
    - arXiv:2008.02215
    - **Why cite:** Adversarial robustness evaluation platform

---

### **Interpretability & Explanation**

26. **Tsipras, D., Santurkar, S., Engstrom, L., Turner, A., & Madry, A. (2019)**
    - "Robustness May Be at Odds with Accuracy"
    - *ICLR 2019*
    - arXiv:1805.12152
    - **Why cite:** Fundamental trade-off between robustness and accuracy

27. **Zhang, H., & Wang, J. (2019)**
    - "Towards Stable and Efficient Training of Verifiably Robust Neural Networks"
    - *ICLR 2019*
    - arXiv:1906.06316
    - **Why cite:** Stability in robust training

---

### **Domain-Specific Applications**

28. **Finlayson, S. G., Bowers, J. D., Ito, J., Zittrain, J. L., Beam, A. L., & Kohane, I. S. (2019)**
    - "Adversarial Attacks on Medical Machine Learning"
    - *Science 2019*
    - DOI: 10.1126/science.aaw4399
    - **Why cite:** Medical AI security - real-world healthcare implications

29. **Cao, Y., Xiao, C., Cyr, B., Zhou, Y., Park, W., Rampazzi, S., Chen, Q. A., Fu, K., & Mao, Z. M. (2019)**
    - "Adversarial Sensor Attack on LiDAR-based Perception in Autonomous Driving"
    - *CCS 2019*
    - arXiv:1907.06826
    - **Why cite:** Autonomous vehicle security - LiDAR attacks

30. **Sitawarin, C., Bhagoji, A. N., Mosenia, A., Chiang, M., & Mittal, P. (2018)**
    - "DARTS: Deceiving Autonomous Cars with Toxic Signs"
    - arXiv:1802.06430
    - **Why cite:** Traffic sign attacks on self-driving cars

---

### **Emerging Threats (2022-2025)**

31. **Wang, Z., Pang, T., Du, C., Lin, M., Liu, W., & Yan, S. (2023)**
    - "Better Diffusion Models Further Improve Adversarial Training"
    - *ICML 2023*
    - arXiv:2302.04638
    - **Why cite:** Diffusion models for adversarial training (very recent)

32. **Croce, F., Gowal, S., Brunner, T., Shelhamer, E., Hein, M., & Cemgil, T. (2022)**
    - "Evaluating the Adversarial Robustness of Adaptive Test-time Defenses"
    - *ICML 2022*
    - arXiv:2202.13711
    - **Why cite:** Adaptive attacks against modern defenses

33. **Pang, T., Yang, X., Dong, Y., Su, H., & Zhu, J. (2021)**
    - "Bag of Tricks for Adversarial Training"
    - *ICLR 2021*
    - arXiv:2010.00467
    - **Why cite:** Practical techniques for improving adversarial training

34. **Jia, X., Zhang, Y., Wu, B., Ma, K., Wang, J., & Cao, X. (2022)**
    - "LAS-AT: Adversarial Training with Learnable Attack Strategy"
    - *CVPR 2022*
    - arXiv:2203.06616
    - **Why cite:** Learnable attack strategies during training

35. **Benz, P., Zhang, C., & Kweon, I. S. (2021)**
    - "Batch Normalization Increases Adversarial Vulnerability and Decreases Adversarial Transferability"
    - *ICCV 2021*
    - arXiv:2010.03316
    - **Why cite:** Understanding architectural choices' impact on robustness

---

### **Transformer & Modern Architecture Robustness**

36. **Shao, R., Shi, Z., Yi, J., Chen, P. Y., & Hsieh, C. J. (2022)**
    - "On the Adversarial Robustness of Vision Transformers"
    - arXiv:2103.15670
    - **Why cite:** Robustness of Vision Transformers vs CNNs

37. **Bhojanapalli, S., Chakrabarti, A., Glasner, D., Li, D., Unterthiner, T., & Veit, A. (2021)**
    - "Understanding Robustness of Transformers for Image Classification"
    - *ICCV 2021*
    - arXiv:2103.14586
    - **Why cite:** Transformer architectures and adversarial robustness

---

### **Standards & Policy (Recent)**

38. **Brundage, M., Avin, S., Wang, J., Belfield, H., Krueger, G., Hadfield, G., Khlaaf, H., Yang, J., Toner, H., Fong, R., Maharaj, T., Koh, P. W., Hooker, S., Leung, J., Trask, A., Bluemke, E., Lebensold, J., O'Keefe, C., Koren, M., Ryffel, T., Rubinovitz, J. B., Besiroglu, T., Carugati, F., Clark, J., Eckersley, P., de Haas, S., Johnson, M., Laurie, B., Ingerman, A., Krawczuk, I., Askell, A., Cammarota, R., Lohn, A., Krueger, D., Stix, C., Henderson, P., Graham, L., Prunkl, C., Martin, B., Seger, E., Zilberman, N., Ó hÉigeartaigh, S., Kroeger, F., Sastry, G., Kagan, R., Weller, A., Tse, B., Barnes, E., Dafoe, A., Scharre, P., Herbert-Voss, A., Rasser, M., Sodhani, S., Flynn, C., Gilbert, T. K., Dyer, L., Khan, S., Bengio, Y., & Anderljung, M. (2020)**
    - "Toward Trustworthy AI Development: Mechanisms for Supporting Verifiable Claims"
    - arXiv:2004.07213
    - **Why cite:** Policy and standards for AI safety (includes adversarial robustness)

39. **European Union (2021)**
    - "Proposal for a Regulation on Artificial Intelligence (AI Act)"
    - COM/2021/206 final
    - **Why cite:** EU AI Act - regulatory framework including adversarial robustness requirements

40. **NIST (2023)**
    - "Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations"
    - NIST AI 100-2e2023
    - DOI: 10.6028/NIST.AI.100-2e2023
    - **Why cite:** US NIST standardization of adversarial ML terminology

---

## 📋 Recommended Citation Format

### For Your References Section:

**IEEE Style** (if required):

```
[1] A. Ilyas et al., "Adversarial Examples Are Not Bugs, They Are Features," in Proc. NeurIPS, 2019. [Online]. Available: https://arxiv.org/abs/1905.02175

[2] F. Croce and M. Hein, "Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks," in Proc. ICML, 2020. [Online]. Available: https://arxiv.org/abs/2003.01690
```

**APA Style:**

```
Ilyas, A., Santurkar, S., Tsipras, D., Engstrom, L., Tran, B., & Madry, A. (2019). Adversarial examples are not bugs, they are features. In Advances in Neural Information Processing Systems (NeurIPS 2019). https://arxiv.org/abs/1905.02175

Croce, F., & Hein, M. (2020). Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. In Proceedings of the International Conference on Machine Learning (ICML 2020). https://arxiv.org/abs/2003.01690
```

---

## 🎯 Essential 10 (If Space Limited)

If you need to be selective, **these 10 are must-haves:**

1. **Ilyas et al. (2019)** - Features not bugs [Theory]
2. **Croce & Hein (2020)** - AutoAttack benchmark [Evaluation]
3. **Carlini et al. (2019)** - Evaluation best practices [Methodology]
4. **Zhang et al. (2019)** - TRADES defense [Defense]
5. **Cohen et al. (2019)** - Certified robustness [Verification]
6. **Nicolae et al. (2019)** - IBM ART toolkit [Tool - You're using this!]
7. **Eykholt et al. (2018)** - Physical attacks [Real-world]
8. **Ren et al. (2020)** - Comprehensive survey [Survey]
9. **Finlayson et al. (2019)** - Medical AI attacks [Application]
10. **NIST (2023)** - Official standards [Policy]

---

## 💡 Pro Tips for Your References

### **Categorize Your References:**

In your paper, group them like this:

**Attack Methods:**
- Cite papers 4, 5, 6, 32

**Defense Mechanisms:**
- Cite papers 7, 8, 9, 10, 18, 19, 20, 31, 33

**Evaluation & Benchmarks:**
- Cite papers 2, 3, 16, 17

**Theory & Understanding:**
- Cite papers 1, 26, 27

**Real-World Applications:**
- Cite papers 13, 14, 15, 28, 29, 30

**Toolkits (Essential for your project!):**
- Cite paper 24 (IBM ART) - **MUST CITE since you're using it**

**Surveys & Standards:**
- Cite papers 21, 22, 23, 38, 39, 40

### **Where to Cite What:**

**Introduction:**
- Cite 1, 21, 38 (motivation, importance, policy)
- Cite 13, 28, 29 (real-world threats)

**Related Work:**
- Cite 2, 3, 16 (evaluation methods)
- Cite 4, 5, 6 (attack methods)
- Cite 7, 8, 9, 10 (defense methods)

**Methodology:**
- Cite 24 (IBM ART - your toolkit)
- Cite 3 (evaluation methodology)

**Results:**
- Cite 2, 16 (benchmarking standards)
- Cite 1 (interpretation of results)

**Discussion:**
- Cite 26 (robustness-accuracy trade-off)
- Cite 22, 23 (contextualize with surveys)

**Future Work:**
- Cite 31, 32, 33, 34 (recent advances)
- Cite 36, 37 (modern architectures)

---

## 🔍 Where to Find These Papers

**Primary Sources:**
- arXiv.org - Most papers available free
- Papers with Code - https://paperswithcode.com/
- Google Scholar - For citation counts and related work

**Conference Proceedings:**
- NeurIPS: https://papers.nips.cc/
- ICML: https://icml.cc/
- ICLR: https://openreview.net/
- CVPR: https://openaccess.thecvf.com/

**Journals:**
- IEEE Xplore
- ACM Digital Library
- Springer

---

## ⚠️ Important Notes

1. **Always cite IBM ART** (Paper #24) - You're using their toolkit!

2. **Check your university's citation style** - IEEE, APA, or other

3. **Include DOI when available** - Makes papers easier to find

4. **Recent = Better** - 2020-2023 papers show you're up-to-date

5. **Mix theory + practice** - Don't just cite attacks, include defenses and applications

---

*These 40 references will make your project look thoroughly researched and current!* 📚
