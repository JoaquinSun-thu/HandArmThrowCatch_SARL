# To all reviewers: Thank you for your comments and we are looking forward to your feedback

Dear reviewers, we would like to thank you for reading and reviewing our manuscript, and your comments did help us improve the manuscript in a more scientific manner. Our reply is presented below, and we would appreciate it if you can take time to read it carefully. Specifically, we have addressed the major concerns of the reviewers by making the following three main changes:
1.We have added a section in Appendix G that introduces the key challenges and solutions related to sim-to-real transfer.
2.We have supplemented our work with additional experiments to demonstrate the robustness of our method under observation disturbances and variations in hand poses.
3.We have clarified the differences between our work and previous research, particularly regarding the Lyapunov stability condition and specific experimental settings.
Furthermore, we hope that the revised manuscript can facilitate reviewers to better understand our work. If you have any more questions or need help understanding anything, please feel free to ask.




# Metareview: 
We would like to thank you for handling our paper. We are also indebted to you and the anonymous reviewers for many helpful comments.


> 
Reviewers appreciate that the paper tackles the problem of dynamic catching, which is an important yet understudied problem in robotics. They found the proposed method clever and interesting. However, reviewers are concerned whether this can be realized in the real world -- especially with the unrealistic motions it generates. A real-world experiment would resolve these concerns.
审稿人赞赏该论文解决了动态捕捉问题，这是机器人技术中一个重要但未得到充分研究的问题。他们发现所提出的方法既聪明又有趣。然而，审稿人担心这是否可以在现实世界中实现 - 特别是它产生的不切实际的运动。一个真实世界的实验将解决这些问题。
> 
We deeply appreciate the reviewers for their valuable comment and suggestions, which greatly helped us to improve the paper's quality. 
非常理解各位评审对仿真到真实世界迁移可行性的担忧。由于本研究只针对手部与腕部，受限于硬件条件，本工作的真实世界的实验设置十分困难，我们无法在本次实验中提供，但为了保证从仿真环境迁移到真实世界的可行性，我们对仿真过程的观测量获得、动作输出以及各种随机性进行了充分的补充测试。我们也对不同任务额外补充了10种unseen物体的实验测试更新了成功率,同时补充了3种更一般的双手姿势进行训练与测试验证,此外我们对机器人throwing and catching这一困难任务分为手部（含腕部）末端，手部结合机械臂，手部结合机械臂与移动基座三个研究阶段，本文为第一阶段，后两个阶段的工作我们也在推进并对第一阶段的工作进行了验证，我们将附上部分结果来支持本工作的可行性。我们已在原文与附录中进行了修改与补充，并提交了补充材料作为附件。
We trust that all of their comments have been addressed accordingly in a revised manuscript.




# Reviewer 1: 
Weaknesses:
> **1**: Further discussion is needed regarding the effectiveness of the trained policies in more general hand posture tasks (beyond those hand postures included in the 5 throwing-catching scenarios mentioned in Line111).

Thank you for your comments. 我们认为本工作设计的五种手部姿态已经包含了双手xyz三种轴向的主要组合，其他手部姿势均可以由这些手部姿势变化而来，为了确保我们策略的有效性，我们进行了两方面补充。
1>我们补充添加了三种更为一般的姿态：overarmout45, abreastin45, overarmrightleft45, 训练后在测试集与训练集上取得较好的meta成功率，见下表。三种姿态的示意图请参见Appendix G.
| | overarmout45| abreastin45| overarmrightleft45|
|:-:|:-:|:-:|:-:
|Train |80.00 | 78.00 | 86.00 |
|Test |80.00 | 78.00 | 86.00 |

2>我们对当前手部姿势的三个轴向均添加了±5°的姿势扰动，同时对双手距离也添加了+-5cm的扰动变化，确保对各种手部姿势的可行性。此外考虑迁移到实际环境的观测误差，我们对物体与目标的位置观测量和姿态观测量分别添加了±5cm和三轴向±20°的观测误差，对物体速度观测量添加了±20%的观测误差。添加上述所有误差后测得的成功率与原成功率对比如下表。
| | Overarm| Abreast| Underarm| Overarm2Abreast|Under2Overarm|Overarmout45| Abreastin45| Overarmrightleft45|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Orignal |80.00 | 78.00 | 86.00 |
|Add random perturbations |80.00 | 78.00 | 86.00 |

对本工作在真实环境中的可行性论证的详细内容已添加至Appendix G。有关视频也已添加至附件文件中.请您查阅.

> **2**: The discussion is insufficient regarding sim2real transfer and challenges that might be encountered. As the state space consists of the information (such as the position of the object, velocities of the object, etc.) that is easily available in the simulator but not in the real world, the method’s strength won’t be adequately established until evaluated against a full transfer to the real word.

Thank you for your comments. 关于您所提到的仿真向真实世界的迁移可能遇到的挑战,我们进行了详细的考虑.
1>我们声明算法框架的所有观测量(状态量)均可以在真实世界中获取.我们已有相关工作验证了输入主视角的RGB图像,利用GPT-4V等多模态大模型,结合yolo[1]等目标检测和SoM[2]的目标分割,可以获取当前物体和目标的初始像素位置,通过6D位姿估计方法Foundation_pose[3]可以实时获取物体过程姿态与像素位置,结合双目相机或深度相机即可得到物体的三维世界坐标,进而计算得到物体的运动速度.物体点云特征是标准化尺寸后仅描述形状特征的先验信息,不需要在过程中实时获取,在无法获得时可以由Appendix B中三类物体的代表几何体代替.双手与关节相关的位姿态速度力等观测量均可由机器人本体获得.指尖的力可以通过指尖力传感器获得.考虑到真实世界观测量获得的观测误差我们进行了抗扰测试,在weaknesses 1中进行了说明.
2>考虑到仿真环境物理性质的偏差,我们算法框架所有动作量减少了力依赖,双手手指关节均为位置控制,只对腕部进行力与力矩控制.在腕部后连接机械臂向真实世界迁移时,可以通过阻抗控制器进行机械臂的规划解算.
3>在本任务中throwing and catching 的物体均为刚体,我们进行了40000+面数的碰撞体构建保证其碰撞体的有效性.同时环境力例如空气阻力等对实验的影响不大，可以忽略.

Reference:
[1] Reis D, Kupec J, Hong J, et al. Real-time flying object detection with YOLOv8[J]. arXiv preprint arXiv:2305.09972, 2023.
[2] Yang J, Zhang H, Li F, et al. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v[J]. arXiv preprint arXiv:2310.11441, 2023.
[3] Wen B, Yang W, Kautz J, et al. Foundationpose: Unified 6d pose estimation and tracking of novel objects[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 17868-17879.


Questions
> **1**: It is stated in Line 250 that “…use the average reward threshold to determine whether it is successful.” However, in Line415 and Line416, one of the components of the reward function is a constant reward r_succ which is given for successfully completing tasks. It seems that there is a loop that how to determine whether it is successful is based on the reward while the reward is related to whether it is successful. Please explain. 

感谢您的问题.过程奖励与测试结果并不存在循环,我们的持续奖励项r_succ is determined by the distance threshold of the final object position from the target position. 而最终测试的成功率判定是通过奖励阈值的.

> **2**: And I would like to know whether r_succ is determined by the distance threshold of the final object position from the target position. If it were, what’s the meaning of using the average reward threshold instead of the distance threshold of the final state from the target position as the success rate (mentioned in Lines 251-253). Since the reward is related to the distance threshold, it seems that there is a redundancy as I’m not sure whether or not the average reward threshold provides more information about the task completion than the distance threshold. Please explain.

正如上一个问题所述,持续奖励项r_succ is determined by the distance threshold of the final object position from the target position. 
这些并不冗余,本工作期望引导双手按照两阶段完成任务.首先完成物体的throwing and catching task, 之后可以已稳定姿态持续的接着物体并在目标附近保持一段时间.仅仅依靠距离阈值做成功率的判断显然是不够的,会出现目标以不稳定姿态接到物体但无法保持的问题,如fig.3.我们的方法设置r_succ是为了引导双手快速完成第一阶段的任务,该判定的奖励尺度与其余奖励尺度相差不大.因此最终结果使用平均奖励阈值可以提供包括抓取物体姿态稳定性与可持续性等更丰富的信息.


> **3**：Relevant but unlikely to deploy to hardware in near future.

关于您对实物验证的担忧,我们已经在推进后续与机械臂结合的下一步工作,并期望在实机上部署,有关内容作为附件附上.


# Reviewer 2: 
Review:

Weaknesses

> **1**: The generalization ability is not tested regarding the distance between the two hands.

Thanks for your comments. 我们对当前手部姿势的三个轴向均添加了±5°的姿势扰动，同时对双手距离也添加了+-5cm的扰动变化，确保对各种手部姿势的可行性。此外考虑迁移到实际环境的观测误差，我们对物体与目标的位置观测量和姿态观测量分别添加了±5cm和三轴向±20°的观测误差，对物体速度观测量添加了±20%的观测误差。添加上述所有误差后测得的成功率与原成功率对比如下表。
| | Overarm| Abreast| Underarm| Overarm2Abreast|Under2Overarm|Overarmout45| Abreastin45| Overarmrightleft45|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Orignal |80.00 | 78.00 | 86.00 |
|Add random perturbations |80.00 | 78.00 | 86.00 |

对本工作在真实环境中的可行性论证的详细内容已添加至Appendix G 并标红。有关视频也已添加至附件文件中.请您查阅.


> **2**: Dynamic tasks are susceptible to external factors in the real world, such as air drag. The robustness of the proposed framework in real-world tasks is unclear.

Thank you for your detailed comments. 
关于您所提到的仿真向真实世界的迁移可能遇到的挑战,我们进行了详细的考虑.
1>我们声明算法框架的所有观测量(状态量)均可以在真实世界中获取.我们已有相关工作验证了输入主视角的RGB图像,利用GPT-4V等多模态大模型,结合yolo[1]等目标检测和SoM[2]的目标分割,可以获取当前物体和目标的初始像素位置,通过6D位姿估计方法Foundation_pose[3]可以实时获取物体过程姿态与像素位置,结合双目相机或深度相机即可得到物体的三维世界坐标,进而计算得到物体的运动速度.物体点云特征是标准化尺寸后仅描述形状特征的先验信息,不需要在过程中实时获取,在无法获得时可以由Appendix B中三类物体的代表几何体代替.双手与关节相关的位姿态速度力等观测量均可由机器人本体获得.指尖的力可以通过指尖力传感器获得.考虑到真实世界观测量获得的观测误差我们进行了抗扰测试,在weaknesses 1中进行了说明.
2>考虑到仿真环境物理性质的偏差,我们算法框架所有动作量减少了力依赖,双手手指关节均为位置控制,只对腕部进行力与力矩控制.在腕部后连接机械臂向真实世界迁移时,可以通过阻抗控制器进行机械臂的规划解算.
3>在本任务中throwing and catching 的物体均为刚体,我们进行了40000+面数凸包的碰撞体构建保证其碰撞体的有效性.同时环境力例如空气阻力等对刚体实验的影响很小,可以忽略.
对本工作在真实环境中的可行性论证的详细内容已添加至Appendix G 并标红。有关视频也已添加至附件文件中.请您查阅.

Reference:
[1] Reis D, Kupec J, Hong J, et al. Real-time flying object detection with YOLOv8[J]. arXiv preprint arXiv:2305.09972, 2023.
[2] Yang J, Zhang H, Li F, et al. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v[J]. arXiv preprint arXiv:2310.11441, 2023.
[3] Wen B, Yang W, Kautz J, et al. Foundationpose: Unified 6d pose estimation and tracking of novel objects[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 17868-17879.




Questions
> **1**: While the experiment results clearly show that the proposed method has better training efficiency compared to the baselines, there does not seem to be a quantified comparison between the success rates of different methods.

Thank you for your detailed comments. 我们已经添加对包括基线的其他方法的成功率测试对比,我们的方法成功率显著优越.相关内容已经在Appendix H中进行了补充添加.
| | Overarm| Abreast| Underarm| Overarm2Abreast|Under2Overarm|Overarmout45| Abreastin45| Overarmrightleft45|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Orignal |80.00 | 78.00 | 86.00 |
|Add random perturbations |80.00 | 78.00 | 86.00 |

> **2**: More settings in the simulation can be changed during experiments to test generalization ability, such as the distance between the hands.

感谢您的建议.正如在weaknesses 1,2中我们提到的,我们充分考虑了仿真到真实世界的迁移可能遇到的问题,进而添加了包括双手距离误差在内的各种扰动来测试策略的鲁棒性.
对本工作在真实环境中的可行性论证的详细内容已添加至Appendix G 并标红。有关视频也已添加至附件文件中.请您查阅.



# Reviewer 3: 
Weaknesses
> **1**: The novelty of the method is limited. It looks like a mix of PPO and Lyapunov functions.

Thank you very much for the question. First, our algorithm is not merely a combination of PPO and Lyapunov functions.
Regarding the Lyapunov component, we do not directly employ the classical Lyapunov constraints for stability requirements. This is because the classical constraint, $L_fV_L(s_t) < 0$, tends to cause the Lyapunov function's value to stagnate at a minimal value close to zero. This drawback impedes policy improvement due to the lack of an explicit gradient.
To address this, we designed a stricter constraint: $L_{f_{\pi_\theta}}V_L(s_t) < -k V^{\alpha}_L(s_t)$, where $\alpha$ and $k$ are constants within the range of 0 to 1. Notably, $k V^{\alpha}_L(s_t)$ is non-negative. This constraint prevents the Lyapunov function from settling at very small values, thereby accelerating the convergence of the stability component in policy improvement. An intuitive illustration can be found in Appendix D, and the experimental results are summarized as follows.
 
| | Overarm| Overarm2Abreast| Under2Overarm| Abreast | Underarm|
|:-:|:-:|:-:|:-:|:-:|:-:|
|PPO with Our Lyapunov design |20.54 | 20.15 | 30.09 | 35.44 | 13.66
|PPO with Standard Lyapunov | 17.15 | 20.12 | 30.11 | 32.22 | 13.60

Furthermore, while the effectiveness of the stability condition has primarily been validated for robustness in previous studies [1-4], this is the first instance where it has been shown to enhance the generation of stable catching behavior for flying objects, thereby improving the accuracy of the throwing-catching process. The results presented in Figure 5, along with the intuitive analysis in Figure 3, illustrate the specific performance and underlying reasons.


Reference:
[1] Westenbroek, T., Castaneda, F., Agrawal, A., Sastry, S., & Sreenath, K. (2022). Lyapunov design for robust and efficient robotic reinforcement learning. arXiv preprint arXiv:2208.06721.
[2] Wang, S., Fengb, L., Zheng, X., Cao, Y., Oseni, O. O., Xu, H., ... & Gao, Y. (2023, December). A Policy Optimization Method Towards Optimal-time Stability. In Conference on Robot Learning (pp. 1154-1182). PMLR.
[3] Chang, Y. C., & Gao, S. (2021, May). Stabilizing neural control using self-learned almost lyapunov critics. In 2021 IEEE International Conference on Robotics and Automation (ICRA) (pp. 1803-1809). IEEE.
[4] Han, M., Tian, Y., Zhang, L., Wang, J., & Pan, W. (2021). Reinforcement learning control of constrained dynamic systems with uniformly ultimate boundedness stability guarantee. Automatica, 129, 109689.





> **2**: There are no real-world results, which I believe matter a lot for RL papers in simulation since anything can happen in simulation and RL can always figure something out. The sim2real gap is large for such dynamic and contact-rich tasks. It would be good if you can show sim2real results or provide a more exhaustive list of sim2real challenges and plans to address them.

Thank you for your comments.您的担忧十分有意义,由于本研究只针对手部与腕部，受限于硬件条件，本工作的真实世界的实验设置十分困难，我们无法在本次实验中提供，但为了保证从仿真环境迁移到真实世界的可行性，我们对sim2real过程可能存在的挑战以及解决方案进行分点说明.
1>挑战:仿真过程中的观测量获得在真实世界中是否可以有效获得.
解决方案:
我们声明算法框架的所有观测量(状态量)均可以在真实世界中获取.我们已有相关工作验证了输入主视角的RGB图像,利用GPT-4V等多模态大模型,结合yolo[1]等目标检测和SoM[2]的目标分割,可以获取当前物体和目标的初始像素位置,通过6D位姿估计方法Foundation_pose[3]可以实时获取物体过程姿态与像素位置,结合双目相机或深度相机即可得到物体的三维世界坐标,进而计算得到物体的运动速度.物体点云特征是标准化尺寸后仅描述形状特征的先验信息,不需要在过程中实时获取,在无法获得时可以由Appendix B中三类物体的代表几何体代替.双手与关节相关的位姿态速度力等观测量均可由机器人本体获得.指尖的力可以通过指尖力传感器获得.
2>挑战:真实世界中获得的观测量存在观测误差如何解决.
解决方案:我们对当前手部姿势的三个轴向均添加了±5°的姿势扰动，同时对双手距离也添加了+-5cm的扰动变化，用于验证策略对各种手部位置姿态的容错性。此外考虑迁移到实际环境的观测误差，我们对物体与目标的位置观测量和姿态观测量分别添加了±5cm和三轴向±20°的观测误差，对物体速度观测量添加了±20%的观测误差,这些误差冗余量均高于当前对应sota方法的误差大小[1-3]。添加上述所有误差后测得的成功率与原成功率对比如下表。
| | Overarm| Abreast| Underarm| Overarm2Abreast|Under2Overarm|Overarmout45| Abreastin45| Overarmrightleft45|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Orignal |80.00 | 78.00 | 86.00 |
|Add random perturbations |80.00 | 78.00 | 86.00 |

3>挑战:仿真过程中的动作量输出在真实世界的控制是否可以实现,是否便于控制,是否可拓展.
解决方案:考虑到仿真环境物理性质的偏差,我们算法框架所有动作量减少了力依赖,双手手指关节均为位置控制,只对腕部进行力与力矩控制.在腕部后连接机械臂向真实世界迁移时,可以通过阻抗控制器进行机械臂的规划解算.因此具备后续将双手腕部后连接机械臂任务的拓展研究与验证.
4>挑战:该任务是否具备可拓展性,是否有后续研究支持验证真实世界效果.
解决方案:我们将机器人throwing and catching这一困难任务分为手部（含腕部）末端，手部结合机械臂，手部结合机械臂与移动基座三个研究阶段，本文为第一阶段，后两个阶段的工作我们也在推进并对第一阶段的工作进行了验证，我们将附上部分结果来支持本工作的可行性。

5>挑战:对于各种unseen物体的有效性泛化.
解决方案:我们的测试实验新补充了10个unseen物体,并将实验结果进行了更新.

6>挑战:对于变化较大的不同手部姿态的有效性泛化.
解决方案:本工作设计的五种手部姿态已经包含了双手xyz三种轴向的主要组合，其他手部姿势均可以由这些手部姿势变化而来，为了确保我们策略的有效性，我们补充添加了三种更为一般的姿态：overarmout45, abreastin45, overarmrightleft45, 训练后在测试集与训练集上取得较好的meta成功率，见下表。
| | overarmout45| abreastin45| overarmrightleft45|
|:-:|:-:|:-:|:-:
|Train |80.00 | 78.00 | 86.00 |
|Test |80.00 | 78.00 | 86.00 |

对本工作在真实环境中的可行性论证的详细内容已添加至Appendix G 并标红。有关视频也已添加至附件文件中.请您查阅.

Reference:
[1] Reis D, Kupec J, Hong J, et al. Real-time flying object detection with YOLOv8[J]. arXiv preprint arXiv:2305.09972, 2023.
[2] Yang J, Zhang H, Li F, et al. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v[J]. arXiv preprint arXiv:2310.11441, 2023.
[3] Wen B, Yang W, Kautz J, et al. Foundationpose: Unified 6d pose estimation and tracking of novel objects[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 17868-17879.


> **3**: The quality of the figures and the video can be improved. For example, the text in the figure is too small and the black background makes it hard to tell the robot hand and the object.

感谢您的建议.我们已经对图中文本小等问题进行了调整和修改,在效果展示图中,为了明确展示手的各个关节,我们将双手各关节设置为了彩色,并对物体添加了色彩渲染,对比使用其他颜色背景后黑色效果相对较好,因此保留.

> **4**: The throwing motion looks unrealistic, e.g., a 720-degree rotation before releasing the object. Hard for me to imagine this working on a real robot. I understand the necessity of such motion to create momentum, but maybe it's more realistic to connect the hand to a robot arm and use the arm motion to create momentum. Connecting the hand to a robot arm in the simulation also adds additional kinematic constraints to the policy to ensure realistic motions.

非常感谢您的建议.
有必要澄清,可能由于文中图片的视角原因给您造成了动作不和实际的误解,我们对策略结果进行了反复检查,双手存在投掷物体后的多余旋转,但并未发现在投掷物体前的不合理动作,我们在附件附上了效果视频以佐证这点.
在双手腕部后连接机械臂的方式十分有效,事实上我们已经在下一步工作中这样实现,正如weakness 2 中我们提到的,我们将将机器人throwing and catching这一困难任务分为手部（含腕部）末端，手部结合机械臂，手部结合机械臂与移动基座三个研究阶段，本文为第一阶段.后续研究的部分效果将作为验证视频附上.

> **5**: The success rate for the training set objects is unsatisfying, although you can argue that the task is very difficult.

感谢您的问题.训练集成功率相对较低由于我们选择了部分代表性的困难物体例如马克笔,对任务成功率有影响.为了更好的验证我们方法的zero-shot的能力,我们对每个任务的测试集又额外添加了10种没见过的物体,最终的成功率更新后已在原文中表1中进行修订并标红.
| | Overarm| Abreast| Underarm| Overarm2Abreast|Under2Overarm|Overarmout45| Abreastin45| Overarmrightleft45|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Orignal |80.00 | 78.00 | 86.00 |
|Add 10 new objects |80.00 | 78.00 | 86.00 |


> **6**: Regarding the plan for real-world experiments, I’m a bit concerned about the embodiment gap between the allegro hand and the shadow hand.

感谢您的建议,在我们后续工作中已经验证了allegro hand 是和shadow hand同样有效的.效果视频在附件中附上.

Questions for Rebuttal:
> **1**: Does the action space include the wrist pose for both hands?

动作空间不包含双手腕部的姿势,仅包含双手腕部的力与力矩.状态空间中包含双手的位置姿态线速度角速度.

> **2**: What’s the current and target object pose on line 123? How do you get the target object pose?

当前物体的位置姿态指准备操作物体的世界坐标位置和他的三轴姿态,目标指希望物体到达的位置与三轴姿态.正如weakness 2 中的介绍, 在真实世界中获取.我们已有相关工作验证了输入主视角的RGB图像,利用GPT-4V等多模态大模型,结合yolo[reference link]等目标检测和SoM[reference link]的目标分割,可以获取当前物体和目标的初始像素位置,通过6D位姿估计方法Foundation_pose[reference link]可以实时获取物体过程姿态与像素位置,结合双目相机或深度相机即可得到物体的三维世界坐标,进而计算得到物体的运动速度.在仿真环境中我们可以直接获得对应物体的位置姿态, 为了模拟现实观测的误差,我们对对物体与目标的位置观测量和姿态观测量分别添加了±5cm和三轴向±20°的观测误差，对物体速度观测量添加了±20%的观测误差,这些误差冗余量均高于当前对应sota方法的误差大小[reference link]。

> **3**: In Section 6.1, using the mean episode reward as the only metric seems not very satisfying. Can you include e.g. the success rate of the policy trained by different networks like in Table 1?

感谢您的建议,我们已经在Appendix H中添加了包括baseline的其他对比实验策略的成功率并标红,请您查阅.

> **4**: How do you decide the reward threshold for the success rate? “Lines 252-253 in Section 6.3 need more explanation.



Thank you for your comments.参考Fig.3的描述,由于稳定点与最优点并不相同,我们通过奖励设计期望引导双手按照两阶段完成任务.首先完成物体的throwing and catching task, 之后可以已稳定姿态持续的接着物体并在目标附近保持一段时间.对奖励阈值的选择,我们通过对测试过程随机抽样50组实验环境,可视化观测其稳定接取物体后对应的奖励值求取平均并进行取整调整后作为奖励阈值.我们已在Appendix A 中进行了补充说明并标红.

> **5**: What does w.d. w.c. w.b. w.a. mean in Figure 5?

Thank you for your question. 这些表示消融实验和baseline相比所添加的不同组件的结果.例如:with a.请见6.2节中对a.b.c.d.四种组件的详细说明. 

> **6**: What would be the main challenges for sim2real transfer? Some things I can think of: How do you acquire the point cloud? Depth cameras in the real world will probably suffer from occluded surfaces, blurred motion, noisy points positions, and features. Will the policy be responsive enough given that it looks like a colored point cloud is needed for the input? Which information is directly available in your state space? How do you estimate the state information for those not directly available?

Thank you for your question. 
物体点云特征是标准化尺寸后仅描述形状特征的先验信息,不需要在过程中实时获取,在无法获得时可以由Appendix B中三类物体的代表几何体代替.
所有状态量均是可观测的,在真实世界均可获得.关于sim2real 迁移的主要挑战和解决方案,已经观测量如何获取等问题我们在weaknesses 2中进行了详细回复,并在Appendix G中进行补充,请您查看. 


> **7**: Contact-rich and dynamic tasks like this usually have very different physics in the real world. Given that you’re using IsaacGym, whose physics I’m not very confident about, why do you think you can do zero-shot transfer?

Thank you for your question.
首先,我们对状态观测量的设计添加了手部姿势,距离,物体位置,姿态,速度,质量,惯量等扰动来保证策略的鲁棒性.其次,我们对动作空间的设计尽量使用了位置控制,仅对腕部使用力控制,降低仿真的影响.此外我们在Appendix B中对大量不同物体进行了聚类分析可以分为三类,我们增加了unseen物体的测试验证确保策略可以支持zero-shot迁移.最后,我们选择的物体均为刚体,我们在仿真中对各种物体进行了100000+体素点,64+凸包的碰撞体构建保证其碰撞体的有效性.我们额外添加了10组unseen新物体做测试验证.通过上述方式,我们任务该方法可以实现zero-shot transfer.



