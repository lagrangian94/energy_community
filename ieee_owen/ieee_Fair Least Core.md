

170IEEE TRANSACTIONS ON ENERGY MARKETS, POLICY, AND REGULATION, VOL. 3, NO. 2, JUNE 2025
Fair Least Core: Efficient, Stable and Unique
Game-Theoretic Reward Allocation in Energy
Communities by Row-Generation
Davide Fioriti, Member, IEEE, Giancarlo Bigi, Antonio Frangioni, Mauro Passacantando,
and Davide Poli
, Member, IEEE
Abstract—Energy Communities are increasingly proposed as a
tool to boost renewable penetration and maximize citizen participa-
tion in energy matters. These policies enable the formation of legal
entities that bring together power system members, enabling collec-
tive investment and operation of energy assets. However, designing
appropriate reward schemes is crucial to fairly foster individuals to
join, as well to ensure collaborative and stable aggregation, maxi-
mizing community benefits. Cooperative Game Theory, emphasiz-
ing coordination among members, has been extensively proposed
for ECs and microgrids; however, it is still perceived as obscure and
difficult to compute due to its exponential computational require-
ments.Thisstudyproposesanovelframeworkforstablefairbenefit
allocation,namedFairLeastCore,thatprovidesuniqueness,repro-
ducibility, stability and fairness. A novel row-generation algorithm
is also proposed that allows to efficiently compute the imputations
for coalitions of practical size. A case study of ECs with up to 100
membersshowsthestability,reproducibility,fairnessandefficiency
properties of proposed model. The results also highlight how the
market power of individual users changes as the community grows
larger, which can steer the development of practical reliable, robust
and fair reward allocations for energy system applications.
Index   Terms—Coalition  fairness  and  stability,  energy
community, energycommunity.jl, fair least core, game theory,
mixed - integer linear programming (MILP).
Received 21 June 2024; revised 14 October 2024; accepted 25 October 2024.
Date of publication 8 November 2024; date of current version 16 June 2025.
This work was supported in part by the European Union–Next-Generation EU
– National Recovery and Resilience Plan (NRRP) – Mission 4, Component 2,
Investment n. 1.1 under Grant call PRIN 2022 D.D. 104 02-02-2022, through
the Project title Large-scale optimization for sustainable and resilient energy
systems, under Grant CUP I53D23002310006 and in part by Investment 1.3 call
for tender n. 1561 of 11.10.2022, under Grant PE0000021, and in part by the
Ministero dell’Università e della Ricerca, CUP concession decree n. 1561 of
11.10.2022 under Grant I53C22001450006, through the Project title Network
4 Energy Sustainable Transition – NEST, Task 8.4.4. Giancarlo Bigi, Antonio
Frangioni, and Mauro Passacantando are members of the Gruppo Nazionale per
l’Analisi Matematica, la Probabilitá e le loro Applicazioni (GNAMPA - National
Group for Mathematical Analysis, Probability and their Applications) of the
Istituto Nazionale di Alta Matematica (INdAM - National Institute of Higher
Mathematics), Piazzale Aldo Moro, 00185 Rome, Italy. Paper no. TEMPR-
00130-2024.(Corresponding author: Davide Fioriti.)
Davide Fioriti and Davide Poli are with the Department of Energy, Systems,
Territory and Construction Engineering, University of Pisa, 56122 Pisa, Italy
(e-mail: davide.fioriti@unipi.it).
Giancarlo Bigi and Antonio Frangioni were with the Dipartimento di Infor-
matica, University of Pisa, 56127 Pisa PI, Italy.
Mauro Passacantando was with the Department of Business and Law, Uni-
versity of Milano-Bicocca, 20126 Milan MI, Italy.
Digital Object Identifier 10.1109/TEMPR.2024.3495237
## I. INTRODUCTION
## A. Motivation
## S
EVERAL governments worldwide[1],[2]are promot-
ing Energy Communities (EC) as a mean to stimulate
investments in renewable assets and increase citizenship par-
ticipation in energy matters. New policies enable the creation
of a legal entity, called “Energy Community”, that aggregates
households, companies and public institutions as members. ECs
are entitled to own and operate energy assets, and promote
the coordination of demand and supply among the members
exploiting them[2]. Hence, there is a pressing need for their
optimal design, taking into account suitable reward schemes
to incentivize member participation. To maximize collective
benefits[3],CooperativeGameTheoryhasbeenextensivelypro-
posed, also in the field of Energy Communities and microgrids.
The Shapley Value has been widely considered the reference
indicator for fairness[4], but suffers from stability concerns[5].
On the other hand, the Core and Nucleolus techniques ensure
stable allocations[6], but not necessarily fair ones. In general,
both approaches are costly from the computational viewpoint,
especially in the planning phase, so that their use in practice
may be challenging. Recent indicators based on convex mea-
sures such as the variance, combined with stability-enforcing
methods, have shown promising results to achieve fair and stable
allocations; yet, computational burden is still a major concern
and uniqueness is not guaranteed[6],[7].
This study proposes novel algorithmic procedures and ef-
ficient implementation techniques to plan the proper design
of Energy Communities and guarantee fair and stable reward
allocation within them.
B. Design of Energy Communities and Aggregators
In recent times, governments worldwide have introduced
supportive policies for renewable energy communities[8]. Their
main target is the promotion of no-profit social, environmental,
and economic targets[3], while meeting the technical chal-
lenges that the energy transition is demanding. Beyond fostering
decarbonization, these initiatives have yielded broad benefits
for power systems, including enhanced reserve provisions[9],
reduced grid congestions[10], increased renewable penetra-
tion[11], and social welfare improvement[2]. However, to fully
© 2024 The Authors. This work is licensed under a Creative Commons Attribution 4.0 License. For more information, see
https://creativecommons.org/licenses/by/4.0/

FIORITI et al.: FAIR LEAST CORE: EFFICIENT, STABLE AND UNIQUE GAME-THEORETIC REWARD ALLOCATION IN ENERGY171
realize these advantages, effective coordination among assets,
consumers, and prosumers is essential. This responsibility falls
on aggregators, tasked with implementing efficient planning
and operation, as well as defining incentive mechanisms that
promote community goals and cohesion.
Traditionally, aggregators are for-profit entities that monitor
and manage the energy system on behalf of consumers and
prosumers. As per EU regulation, aggregators cannot be ECs
themselves, given their for-profit nature[3]. However, they can
have a support role in its creation, management and operation;
for this reason, they can be regarded as a player in the EC and, as
such, they shall be rewarded appropriately, also not to incur in
the so-called agency problem[6]. Their role in energy manage-
ment is better known, although most studies regarding advanced
market structures[12], demand-side management[13],[14]and
stochastic approaches[15]have been generally focused on large
energy players. Furthermore, small and medium consumers,
namely the target of EC policies, have rarely played an active
role in energy markets and even less in providing flexibility
or ancillary services[16]. Previous studies have primarily em-
phasized economically-driven techniques devoted to optimally
operate the aggregate[17], using Mixed-Integer Linear Pro-
gramming (MILP). Some of them have explored maximizing
the aggregate’s social welfare, but may have overlooked the fair
distributionofprofitsamongparticipants[18],[19].Inparticular,
the fair stable reward of aggregators—a critical but complex
topic—has been rarely considered in the literature[6]. For these
reasons, in this study we develop a MILP planning model able to
account for the role of members, including the aggregator, and
their fair and stable reward.
C. Competitive and Cooperative Reward Allocations
In the context of local energy markets and ECs, competitive or
cooperative incentive mechanisms are commonly proposed, yet
generally in the field of operation rather than of planning[12].
In competitive approaches, users operate independently to max-
imize their individual benefits, potentially competing with each
other for scarce common resources[12],[19]. In this case,
there is no guarantee that the solution maximizes the utility
of the aggregate, thus competition can be detrimental. Non-
cooperative strategies, such as those proposed in[20]and[21]
that include operational flexibility, focus on optimizing the
actions of individual aggregators or users in a local energy
market or network. Nash’s theory is widely adopted in this
context to identify the market equilibrium[12],[22],[23].Even
if competition may suit some scenarios, cooperation can be
limited, thus potentially leading to sub-optimal results, which
can oppose the social goal desirable by policies, such as the EU
regulation[8]. Moreover, most non-cooperative techniques rely
on bidding by members[12], an approach whose practicality is
notentirelyconfirmedonthelargescale.IntypicalECs,usersare
less likely to individually perform active trading, and therefore
cooperative approaches are particularly relevant[6],[11].In
these approaches users cooperate towards the best outcome for
the entire community and distribute rewards according to each
individual contribution[19], with no detrimental effect on the
global benefit. The Shapley Value is generally considered the
reference indicator for fair reward sharing in coalition games[4],
[24],[25]. However, it suffers from stability issues, i.e., there is
no guarantee that no subset of users is better off from leaving the
community[26],[27], as proven in[6]inthecontextofthedesign
ECs, including investment costs. Moreover, the computation of
the Shapley value is very demanding (see also SectionIII-G),
and stability concerns remain even with approximations[24].
The center-of-gravity of the imputation-set value (also known
as egalitarian value) and the egalitarian non-separable contri-
bution value, both first introduced in[28]and further analyzed
in[29],[30],[31], are alternative fair allocation methods which
are less computationally demanding, and therefore have found
application, among others, in energy settings[32],[33].The
former assigns to each player an equal share over its individual
worth, while the latter exploits the same idea for the marginal
contribution of each player. However, they cannot be applied
in the EC setting since the peculiar role of the aggregator
cannot be taken into account. Moreover, they have the same
stability issue as the Shapley value. The set of reward allocations
(a.k.a., imputations) that guarantee stability is named Core, and
is typically not a singleton. However, imputations within the
Core may be marginally stable, i.e., a subset may be equally
better off inside or outside the community. For these reasons, the
stricter formulations of the Least Core[15]and Nucleolus[34]
have been proposed: the former maximizes the benefit of the
coalition that is most likely to exit the community, whereas the
latter iteratively applies the same concept to each most likely
coalition to leave. Conversely to Core and Least Core, Nucleolus
is proven to be unique[35], which is a desirable property, but the
corresponding computational approaches are very challenging,
as proven, e.g., in[36]and references therein.
D. Computational Challenges of Game-Theoretic Allocations
Despite its benefits, the combinatorial nature of cooperative
game theory is a significant barrier to its practical use. In case
studies involving a small number of members, their enumerative
formulation can be used[6],[37], but with communities ex-
ceeding 20-30 members the computational requirements quickly
become prohibitive. In[38]an approximation for the Shapley
Value has been proposed that reduces the combinations by about
99%; yet, concerns on stability still apply. Nucleolus and Least
Core have been used in various studies but only for system
operation[27],[39], with no application to ECs. One of the few
exceptionsis[6],butthecomputationalapproachusedtheredoes
not scale efficiently with size. A decomposition algorithm of
Nucleolus based on Benders’ decomposition is proposed in[27],
but it is not applicable in the EC field given the intrinsic binary
nature of membership of each user to the EC. To overcome
that, a simplification of Nucleolus has been proposed using
a pure Variance equivalence[7], but stability concerns were
overlooked. For these reasons, in[6]a methodology is proposed
to stabilize imputation; yet, the approach is still combinatorial
and limited to few members. An alternative solution is offered by
the Owen sharing method[15]that distributes the reward based
on the equivalent market prices created by the dual solution of
the optimization problem for the bidding of wind generators.
However, while being simple to calculate, the Owen solution

172IEEE TRANSACTIONS ON ENERGY MARKETS, POLICY, AND REGULATION, VOL. 3, NO. 2, JUNE 2025
Fig. 1.  Business model of the energy community.
may not achieve desirable properties such as these of Least Core,
Nucleolus or Shapley Value[40]. Row-generation has been
shown to be a promising approach for decomposing Nucleolus-
like formulations[41], among other problems[42], but it has
not been applied to EC. For these reasons, it is considered in
this study and combined with Core, Least Core and Variance
mechanisms.
E. Contributions and Organization of the Paper
The main contributions of our work are as follows:
1) definition of generalized reward allocation schemes,
named Fair Core and Fair Least Core, that aim at max-
imizing fairness and stability of reward allocations;
2) uniqueness and reproducibility for Fair Core and Fair
## Least Core;
3) novel algorithm to efficiently calculate reward alloca-
tion methods for ECs of arbitrary composition, includ-
ing row-generation and smart decomposition of the EC
planning problem that have been implemented in the
open-source packegesEnergyCommunity.jl[43]andThe-
oryOfGames.jl[44];
4) application and validation of the algorithm to several re-
wardallocationmechanisms andcomparisonwithexisting
methodologies, e.g. Shapley Value and Nucleolus;
5) evaluation of the impact of EC size into the fair reward
allocation to provide policy recommendations.
The remainder of the paper is organized as follows. SectionII
describes the EC and its mathematical optimization problem.
SectionIIIreviews the literature about reward allocation by
game theory. SectionIVdetails the general fair stable reward
allocation proposed in this study, whose efficient computation
is detailed in SectionVand SectionVI. The case study and
results are reported in SectionVIIandVIII, respectively. Finally,
conclusions are drawn.
## II. T
## HEENERGYCOMMUNITYPLANNING
## A. Business Model
According to the literature[6]and the European Union reg-
ulation[1], an EC operates as a non-profit entity, sharing all
revenues among its members after fulfilling its obligations;
technical support by third-parties, e.g. aggregators, is admitted.
Accordingly, this study focuses on the business model depicted
in Fig.1, where members create the legal entity Energy Com-
munity and can engage with an aggregator to maximize the
overall benefits. An EC coordinated by an aggregator is denoted
with “CO”. Without the aggregator, the community is still able
to create an EC, referred to as Aggregated Non-Cooperative
(ANC),butitcannotcoordinateconsumptionandproductionnor
theinvestmentstoachievethemaximumeconomicperformance.
The EC is awarded an economic benefit for every unit of energy
that is produced by a user and virtually consumed by another
user in the same time step[16]. Each user has its own energy
provider and can invest in renewable assets or storage, in case
keeping full ownerhip of such devices. In the CO configuration,
users collaborate to maximize the overall benefit measured with
Net Present Value (NPV)[32], and they shall be remunerated
fairly. Finally, in cases where no EC is established, referred
to as the Non-Cooperative (NC) configuration, users invest in
local decentralized resources to maximize their own profits. This
scenario serves as the baseline for the analysis.
We now present bare-bones, yet sufficient for the present dis-
cussion and computational testing, mathematical models for EC
planning and operations. These models are in agreement to the
most recent EU and Italian regulation[16]. We remark that more
sophisticated EC models (including, e.g., operations on different
market, other generation units with more complex operational
constraints, multi-energy aspects, and even the representation of
time-variant dynamics by stochastic approaches) could be used
without significantly impacting the proposed approach or the
computational algorithm, as discussed in details later on. The
modular structure of the developed open-source tools makes it
easy to adapt to such types of model improvements.
## B. Users’ Objective
When no EC is established, the objective of each userjis
to maximize its own NPV, reported in(1), composed of the net
profit for selling/buying electricity to/from the market (R
y
## )for
each yeary∈Y, the investment costs (I
y
) that are non-null only
at the first year (y=0), the operating charges due to peak tariffs
and maintenance charges (OP
y
), the replacement costs of the
assets (RP
y
) and the recovery value (RV
y
), which is non-null
onlyattheendoftheproject.Neteconomicflowswiththeenergy
market are modelled in(2), accounting for users-specific tariffs
that are represented by selling prices (π
## +
j,t
), buying prices (π
## −
j,t
## ),
including grid tariffs and taxes, and excises (π
ex
j,t
) for each time
stept∈T; the weightm
## T
t
accounts for granularity and number
of representative days.P
## U±
t
denotes the power exchanged at
the Point of Delivery (POD), where positive apex stands for
injection into the distribution grid;P
## L
t
is the consumer demand.
According to(4), for every tariff horizonw∈W(for instance
in Italy corresponding to a month), the peak chargesOP
y
are
dynamically accounted for considering the peak tariffc
## P
w
and the
actual maximum power exchangedP
## Umax
w
at the POD. Yearly
maintenance costsOP
y
, represented by the second term of(4),
are proportional to the investment capacityx
a
of each assetaof
the set of assetsA
j
of userj, according to a coefficientc
a,M
## .
Replacement chargesRP
y
detailed in(5)apply when an asset
reaches its end of lifeN
## Y,a
, while the residual value of assets is
recovered as described in(6).rrepresents the discount rate.
## NPV
j
## =
## 
y∈Y
## R
j,y
## −I
j,y
## −OP
j,y
## −RP
j,y
## +RV
j,y
## (1 +r)
y
## (1)
## R
j,y
## =
## 
t
m
## T
t
## 
π
## +
j,t
## P
## U+
j,t
## −π
## −
j,t
## P
## U−
j,t
## −π
ex
j,t
## P
## L
j,t
## 
## (2)

FIORITI et al.: FAIR LEAST CORE: EFFICIENT, STABLE AND UNIQUE GAME-THEORETIC REWARD ALLOCATION IN ENERGY173
## I
j,0
## =
## 
a
c
a,I
j
x
a
j
## (3)
## OP
j,y
## =
## 
w
m
## W
w
c
## P
j,w
## P
## Umax
j,w
## +
## 
a∈A
j
x
a
j
c
a,M
j
## (4)
## RP
j,y
## =
## 
## 
a∈A
j
x
a
j
c
a,I
j
if  mod (y,N
## Y,a
j
## )=0
## 0else
## (5)
## RV
j,|Y|
## =
## 
a∈A
j
x
a
j
c
a,I
j
## N
## Y,a
j
−mod (|Y|−1,N
## Y,a
j
## )
## N
## Y,a
j
## (6)
## C. Constraints
This section details the major technical constraints of each
user. The power balance within each internal system is ensured
through(7), whereP
## U±
j,t
denotes the power dispatch at the user’s
## POD,P
c±
j,t
represents the power dispatch of the battery converter
(with+indicating supply and−indicating absorption),P
## R
j,t
corresponds to the renewable production, andP
## L
j,t
is the demand.
## A
## C
j
denotes the converters of userj.
## P
## U+
j,t
## −P
## U−
j,t
## +
## 
c∈A
## C
j
## 
## P
c−
j,t
## −P
c+
j,t
## 
## −P
## R
j,t
## =−P
## L
j,t
## ∀t(7)
The peak power at the user POD is calculated with(8), where
## ˆ
## T
w
denotes the set of time steps corresponding to the peak
power periodw∈W. Constraint(9)specifies that the renewable
productionP
## R
j,t
of each userjat time steptmust not exceed the
sum (over all renewable technologiesr∈A
## R
j
) of the maximum
available power dispatch, which is proportional to the installed
capacityx
r,U
j
and its specific power productionp
r,U
j,t
## .
## P
## Umax
j,w
## ≥max

## P
## U+
j,
## ˆ
t
## ,P
## U−
j,
## ˆ
t


## ∀w,
## ˆ
t∈
## ˆ
## T
w
## (8)
## P
## R
j,t
## ≤
## 
r∈A
## R
j
p
r,U
j,t
x
r,U
j
## ∀t(9)
The energy balance of the batteries is modeled using(10),
employing cyclical notation (E
b,U
j,0
## =E
b,U
j,|T|
); equations account
for the roundtrip efficiencyη
b
j
of the batteryb, including its cor-
responding converterc=c(b)∈A
## C
j
, belonging to the setA
## B
j
## .
The peak power capacity is ensured by(11), while the maximum
and minimum allowed state of charge are taken into account in
(12)using coefficientsβ
b,max
j
andβ
b,min
j
. The variablesx
b
j
and
x
c(b),U
j
represent the rated energy capacity of batteryband the
power capacity of the corresponding converter, respectively.
## E
b
j,t
## =E
b
j,t−1
## −ΔP
c(b)+
j,t
## /

η
b
j
## +ΔP
c(b)−
j,t

η
b
j
∀b, t(10)
## P
c±
j,t
## ≤x
c,U
j
∀c, t(11)
x
b
j
β
b,min
j
## ≤E
b
j,t
## ≤x
b
j
β
b,max
j
∀b, t(12)
D. Energy Community Objective and Shared Energy
In a Cooperative Energy Community, the overall goal is
to maximize the so-called social welfareSW
## CO
(K)of the
communityK, which includes the NPV of each memberjand
the total rewardR
## SH
y
allocated to the community, as detailed
in(13).ThetotalrewardannuallyawardedtoanECisformulated
in(14), whereπ
## SH
t
is the regulated unitary reward andP
## SH
t
is
the shared energy virtually net-metered.P
## SH
t
is defined as the
minimum between the overall production and consumption, as
modelled in(15).
## SW
## CO
## (K)=
## 
j∈K
## NPV
j
## +
## 
y∈Y
## R
## SH
y
## (1 +r)
y
## (13)
## R
## SH
y
## =
## 
t
π
## SH
t
m
## T
t
## P
## SH
t
## (14)
## P
## SH
t
## =min
## ⎧
## ⎨
## ⎩
## 
j∈K
## P
## U+
j,t
## ,
## 
j∈K
## P
## U−
j,t
## ⎫
## ⎬
## ⎭
## ∀t(15)
## E. Energy Community Problems
1) Coordinated EC Problem (CO):In abstract terms, letu
j
be the operation (P
## U±
j,t
## ,P
## Umax
j,w
## ,P
## R
j,t
## ,P
c±
j,t
## ,E
b
j,t
) and investment
variables (x
a,U
j,t
) of each userj, andsthe power shared in an
EC. The mathematical problem for the coordinated EC is shown
in(16), where matrixM
j
and vectorb
j
denote the constraints
in SectionII-C, while constantsc
j
andl
j
represent the cost
coefficients discussed in SectionII-B. The shared powersis
constrained to be lower than or equal to the total energy pro-
duction and consumption, by using matricesD
## ±
, through the
identityP
## U±
j
## =D
## ±
u
j
;δ>0represents the weighted reward
for every unit of shared power.
## SW
## CO
(K)=  max
## 
j∈K
## (c
## T
j
u
j
## +l
j
## )+δ
## T
s
s.t.M
j
u
j
## ≤b
j
∀j∈K
s≤
## 
j∈K
## D
## +
u
j
s≤
## 
j∈K
## D
## −
u
j
## (16)
This formulation turns out to be useful in the discussion of the
other EC problems described below.
2) Non-Coordinated Users Problem (NC):As discussed in
SectionII-B, in this case each user maximizes its own prof-
itability regardless of the others. LetSW
## NC
(K)be the optimal
objective function of the optimization of the whole community
with no user interaction, as in(17). No shared energy applies
and hence no coordination is incentivized.
## SW
## NC
## (K)=
## 
j∈K
maxc
## T
j
u
j
## +l
j
s.t.M
j
u
j
## ≤b
j
## (17)
It is worth noticing that the problem in(17)is similar to(16),but
no shared energy applies. That indeed leads each user problem
to be independent.
3) Aggregated-Non-Coordinated EC Problem (ANC):Fi-
nally, we consider the so-called Aggregated-Non-Coordinated
EC problem, where users create an EC, but no aggregator is
present to coordinate the operation of the system, nor to rec-
ommend coordinated investments to the users. In this case, the
users are expected to behave as in the NC problem, but also
benefit from the (probably low) shared energy corresponding to
the non-coordinated system operation. Let
u
## NC
j
be the optimal

174IEEE TRANSACTIONS ON ENERGY MARKETS, POLICY, AND REGULATION, VOL. 3, NO. 2, JUNE 2025
decision vector of userjin the NC problem, then the overall ob-
jective function of the whole community under ANC conditions
can be described as in(18):
## SW
## ANC
## (K)=SW
## NC
(K)+maxδ
## T
s
s.t.s≤
## 
j∈K
## D
## +
u
## NC
j
s≤
## 
j∈K
## D
## −
u
## NC
j
## (18)
It is worth noticing that the problem in(18)is similar to(16),
but the decision variablesu
j
are set to the NC optimal solution.
Accordingly, users constraints (M
j
u
## NC
j
## ≤b
j
) are satisfied by
definition of
u
## NC
j
, and hence excluded from the optimization.
## III. G
## AME-THEORETICREWARDALLOCATION
A. Preliminary Definitions: Benefit and Surplus of a Coalition
A cooperative game with transferable utility can be devised to
reward the participants in the EC. The set of playersIis made by
the setJof users that may join the community and the aggregator
A. The characteristic functionvmeasures the common benefit of
the possible ECs between the players who agree to join it even-
tually including the aggregator. Each user can always choose its
own NC optimal solution, therefore this is considered as the base
case configuration. When the AggregatorAparticipates, the CO
optimal solution can be achieved and the corresponding benefit
is the difference between the optimal performance of CO and
NC configurations; otherwise, no coordination is created and the
benefit of the community is limited to the difference between the
optimal ANC and NC configurations.
The mathematical expression of the characteristic function
for any coalitionK⊆Iis given by
v(K)=
## 
## SW
## CO
## (K
## A
## )−SW
## NC
## (K
## A
)ifA∈K,
## SW
## ANC
## (K)−SW
## NC
(K)ifA/∈K,
## (19)
whereK
## A
## =K\{A}.
To identify the improvement of benefit for each user or ag-
gregator in the presence of the community, we consider the set
## B=
## 
## Δ∈R
## |I|
## +
## :
## 
i∈I
## Δ
i
=v(I)
## 
## (20)
which describes the possible ways the overall improvementv(I)
is shared between them. Once an allocationΔ∈Bis chosen,
the improved NPV of each userjwith respect to the base case
(NC) is given by
## NPV
## F
j
## =NPV
## NC
j
## +Δ
j
## .(21)
For ease of presentation, we introduce the concept of surplus
σ(K,Δ)of a coalitionK⊆Iwith respect to allocationΔas
σ(K,Δ) =
## 
i∈K
## Δ
i
−v(K).(22)
Whenσ(K,Δ)is positive, the users are better off within the
community rather than being on their own.
## B. Core
TheCore[45]is the set of reward allocations that guarantees
that no coalition of the whole communityIis worse off within
the community than outside, i.e.,
C(I,v)={Δ∈B:σ(K,Δ)≥0∀K∈P},(23)
whereP={K⊂I:K=∅}is the set of proper subsets of
I. This property ensures the stability of the coalition, in that no
user is expected to benefit from leaving the community. As it
is defined by a finite number of linear inequalities,C(I,v)is a
polytope and may contain uncountably many allocations.
## C. Least Core
TheLeast Core[46]is the set of allocations that maximize
the benefit for the least profitable coalition, i.e.,
LC(I,v)=
## 
Δ∈B:σ(K,Δ)≥θ
## LC
## ∀K∈P
## 
## ,(24)
where
θ
## LC
## =maxθ
s.t.σ(K,Δ)≥θ∀K∈P
## Δ∈B
## (25)
While the Core might be empty, the Least Core is always
nonempty. In particular, ifθ
## LC
<0then the Core is empty.
Otherwise, ifθ
## LC
>0the Least Core is a proper subset of
the Core, while they coincide wheneverθ
## LC
=0. Clearly, the
computational burden ofLCis equivalent to the Core.
## D. Nucleolus
Given any allocationΔ,letψ(Δ)be the order vector of satis-
faction, i.e., the vector of surpluses arranged in non-decreasing
order. The Nucleolus[35]is the unique allocation that lexico-
graphically maximizes the vectorψ. In comparison with core
and least core, Nucleolus is computationally harder to compute.
Indeed, the computation ofθ
## LC
is just the first step of the
lexicographic maximization ofψ.
## E. Shapley Value
TheShapley Valueis the only allocation that jointly satisfies
efficiency, symmetry, dummy, and linearity properties[47].The
allocation of each playeri∈Iis the weighted average of its
marginal contribution to every coalition:
## Δ
## SV
i
## =
## 1
## |I|
## 
## K⊆I
## 
## |I|−1
## |K|
## 
## −1
[v(K)−v(K\{i})].(26)
The Shapley Value may not belong to the Core and it is as
computationally intensive as the Core calculation.
F. Variance Core and Variance Least Core
In order to select an allocation in the Core or Least Core,[6]
proposed to minimize the squared distance from the uniform
allocation. The corresponding unique minima
## Δ
## VC
## =argmin
## 
## 
i∈I
## 
## Δ
i
## −
v(I)
## |I|
## 
## 2
:Δ∈C(I,v)
## 
## (27)

FIORITI et al.: FAIR LEAST CORE: EFFICIENT, STABLE AND UNIQUE GAME-THEORETIC REWARD ALLOCATION IN ENERGY175
## Δ
## VLC
## =argmin
## 
## 
i∈I
## 
## Δ
i
## −
v(I)
## |I|
## 
## 2
:Δ∈LC(I,v)
## 
## (28)
have been namedVariance Core(VC) andVariance Least Core
(VLC). In the authors’ opinion this approach appears promising;
in the following, we generalize its formulation also including
proof of uniqueness.
## G. Computational Concerns
The common computational issues involved in the reward
distributions schemes described above stem from the need of
computing the value ofv(K)for every subsetK⊆I. This in-
volves solving a number of optimization problems, described in
SectionII-E, that is exponential in the size|I|of the community.
Consequently, these models can hardly be used, with a naïve
computation approach, for problems larger than 10-20 users[6];
this has so far limited the use of game theoretical approaches in
ECs and in the power systems field.
The complexity issue clears comes from the fact that the
above formulations involve a number of variables that grows
linearly with the size of the communityI, but a number of
constraints that is exponentially in the number of coalitionsK,
i.e., of the order of2
## |I|
. Yet, it is well-known that in such a case
only a small fraction of the constraints are going to be binding,
i.e., that there exists a formulation with a manageable number
of constraints—corresponding to a small, well-chosen set of
coalitions—that is in fact equivalent to the full one. The issue is
that this set is not known in advance: however,row-generation
approaches have proven able to efficiently solve problems of this
type, provided that a properseparation oraclecan be developed
to efficiently identify constraints (coalitions) violated by a given
solution. In the following, we show how this can be done for a
large class of practical EC models, thereby allowing the actual
use of game-theoretic concepts for community of the scale
required by practical applications.
## IV. F
## AIRCORE ANDFAIRLEASTCORE
## A. Definition
Different measures of fairness rather than variance can be
considered. Therefore, we propose the generalFair Core(FC)
andFair Least Core(FLC) reward allocation schemes in the
same fashion, by maximizing a generic strictly concave function
fthat measures the fairness of allocationΔover the Core or
## Least Core:
## Δ
## FC
=argmax{f(Δ) : Δ∈C(I,v)},(29)
## Δ
## FLC
=argmax{f(Δ) : Δ∈LC(I,v)}.(30)
With respect to the existing approaches, the proposed framework
allows generalizing fairness measures while ensuring the stabil-
ity of the coalition. For example,fcould capture the solution
closest to the egalitarian solution within the [L]C or take into
account social measures such as energy poverty[48].
B. Properties: Existence, Uniqueness, Reproducibility and
## Stability
When the Core is empty,Δ
## FC
is not even defined. On the
contrary,Δ
## FLC
always exists. Moreover, the choice of strict
concavity offguarantees the uniqueness of the optimal solution
of the above problems, see for instance[49], so that(29)and(30)
define unique allocations. Note that VC and VLC are special
cases of FC and FLC, respectively: minimizing variance is
equivalent to maximizing negative variance, that is a (strictly)
concave function. Consequently, repeated tests with different
algorithms applied on FLC will always converge to the same
unique optimal solution. This guarantees that the solution is
reproducible, which is crucial for practical applications as it
helps prevent misinterpretations. Moreover, as F[L]C belongs
to [L]C by definition, then stability is guaranteed[6],[26].
## V. T
## HEPROPOSEDCOMPUTATIONALALGORITHM
## A. The Algorithm
We focus on the solution of problem(30), since it is more
complex than(29)and than the computation of just one point in
the Core and Least Core.
The algorithm is divided into two consecutive stages. The first
aims at computingθ
## LC
together with one point of the Least Core
and, once the former is approximately known, the second stage
actually solves(30). In order to solve(29)the first stage is not
needed, as the Core is nothing else than the Least Core(24)with
θ
## LC
=0. As a consequence, the computation of just a point in
the Core can be performed through the second stage with the
particular choice off=0.
Since the problems in both stages involve an exponential
number of constraints, we propose the use of a row-generation
technique to efficiently deal with them. The overall algorithm is
sketched in Fig.2. Each stage proceeds by iteratively executing
a Master Problem (MP) and a Separation Problem (SP). The
MP generates candidate reward allocations by considering only
the constraints corresponding to a (small) subsetΓof proper
coalitions, that is iteratively revised. Given the optimal solution
of the MP, the SP seeks to find the coalition with the lowest
surplus, that is therefore added to the setΓ.
In the first stage, convergence is reached when the surplus of
MP matches the optimal value of SP. In the second stage, it is
reached when the coalition found by SP is feasible for the MP,
and this happens when the approximated value ofθ
## LC
computed
at the first stage matches the optimal value of SP.
An important aspect to improve the performance of the algo-
rithm is the initialization ofΓwith a well-chosen pre-defined set
of coalitions.
## B. Initialization
The aim of the initialization is to populate the setΓwith a
pre-set, low number of coalitions, for each of which the quantity
v(K)must be computed. While computingv(K)has generally
lower computational requirements with respect to the SP, doing
so an exponential number of times is prohibitive. Pre-populating
## Γhasacostproportionaltothechosensize,butontheotherhand,

176IEEE TRANSACTIONS ON ENERGY MARKETS, POLICY, AND REGULATION, VOL. 3, NO. 2, JUNE 2025
Fig. 2.  Proposed solution algorithm for the fair least core.
a largerΓcan be expected to yield faster convergence. Therefore,
a trade-off exists that will be explored in the computational
section. Besides the number of coalitions, we will show that
their effective choice is crucial.
## C. First Stage
Given the setΓ, the Master Problem is
ω
## M
=max{θ:Δ∈B,σ(K,Δ)≥θ∀K∈Γ},(31)
which is the relaxation of(25)obtained by only considering the
constraints corresponding to the coalitions inΓ.IfthesetΓis
reasonably small, then(31)can be efficiently solved since it has
|I|+1variables in total (θand the allocationΔ). This provides
an optimal allocationΔ
## M
and its valueω
## M
. Since(31)has less
constraints than(25), thenω
## M
## ≥θ
## LC
## .
The Separation Problem checks ifΔ
## M
is actually feasible for
(24)by finding the coalitionK
## S
with lowest surplus
ω
## S
## =min
## 
σ(K,Δ
## M
## ):K∈P
## 
## .(32)
## Ifω
## M
## =ω
## S
, thenθ
## LC
## =ω
## M
andΔ
## M
belongs to the Least
Core. To lower the computational burden, the above equality
between the optimal values is checked up to some desired
precisionε, the approximate value
## ˆ
θ
## LC
## =ω
## M
is exploited in the
second stage and the first stage is considered over. Otherwise,
## K
## S
is added toΓand the Master Problem(31)is solved again.
## D. Second Stage
## Given
## ˆ
θ
## LC
and the setΓprovided by the first stage, the Master
Problem in the second stage is
max

f(Δ) : Δ∈B,σ(K,Δ)≥
## ˆ
θ
## LC
## ∀K∈Γ


## ,(33)
which is an approximation of(30)since
## ˆ
θ
## LC
is kept fixed.
## When
## ˆ
θ
## LC
## =θ
## LC
, any optimal allocationΔ
## M
of(33)provides
an upper boundf(Δ
## M
)of the optimal value of(30)and it is
optimal if it is feasible for(30). Therefore, we stop the second
stage wheneverΔ
## M
is feasible in any case since
## ˆ
θ
## LC
is always
expected to be very close to the true valueθ
## LC
. Feasibility can be
checked by solving the Separation Problem(32)and comparing
ω
## S
with
## ˆ
θ
## LC
. If they are (approximately) equal, thenΔ
## M
is
feasible, otherwise the optimal coalitionK
## S
is added to the set
Γand a new iteration is performed.
## VI. T
## HESEPARATIONPROBLEM
While the MP is a continuous optimization problem, the SP is
combinatorial. Yet, by exploiting the mathematical formulation
for the EC problems of SectionII-E, it can be recast as a MILP,
and therefore solved efficiently for communities of practical
size, as shown in the following.
A. Mapping a Generic Coalition
The crucial challenge is to develop a proper row-generation
algorithm to efficiently solve the Separation Problem(32).This
requires in particular to describe the surplus function defined
in(22)for a generic coalitionK⊆I. The fundamental mod-
elling trick we exploit is to augment the models of SectionII-E
with membership binary variablesz∈{0,1}
## |I|
to represent the
chosen coalition; that is,z
i
equals 1 when memberibelongs to
the coalitionK, and 0 otherwise.
B. Benefit of a Coalition
We now describe how to model the benefitv(K)of a coalition,
defined in(19), for any coalitionKrepresented by the variable
z. Since the presence of the aggregator significantly changes
the structure of the mathematical problem in(19)that must be
solved, we separate the functionvinto the two components,
v
## W
andv
## W/O
, which represent the case with and without the
aggregator, respectively; that is,
v(K)=v(z)=
## 
v
## W
## (z)ifz
## A
## =1,
v
## W/O
## (z)ifz
## A
## =0,
## (34)
wherez
## A
is the membership variable of the aggregator.
1) Coalition With the Aggregator:The benefitv
## W
## (z)repre-
sents the difference between(16)and(17), namely
v
## W
(z)=  max
u,s
## 
j∈J
## (c
## T
j
u
j
## −c
## T
j
## ̄u
## NC
j
z
j
## )+δ
## T
s
s.t.M
j
u
j
## ≤b
j
z
j
∀j∈J
s≤
## 
j∈J
## D
## +
u
j
s≤
## 
j∈J
## D
## −
u
j
## (35)
where the variablesuof all members are formally included
together with the energy exchanges. Anyway their actual
occurrence is driven by the choice of the coalition addressed
byz. In fact, sinceM
j
u
j
## ≤b
j
includes box constraints, zero-
ing the right-hand-side forces all variablesu
j
to be zero, as
## {u
j
## :M
j
u
j
≤0}={0}. This suggest to replaceb
j
withb
j
z
j
## .
## Indeed,choosingz
j
## =0impliesu
j
## =0:memberj“disappears”

FIORITI et al.: FAIR LEAST CORE: EFFICIENT, STABLE AND UNIQUE GAME-THEORETIC REWARD ALLOCATION IN ENERGY177
from the problem and cannot contribute to the energy exchange
variablessand their reward, while on the other hand not in-
curring in any cost. Conversely, whenz
j
=1the constraint
readsM
j
u
j
## ≤b
j
and memberj“operates normally”, thereby
contributing to the community but having to pay its normal
costs.
2) Coalition Without the Aggregator:The benefitv
## W/O
## (z)
is the difference between(18)and(17), which corresponds to
v
## W/O
(z)=  max
s
δ
## T
s
s.t.s≤
## 
j∈J
## D
## +
u
## NC
j
z
j
s≤
## 
j∈J
## D
## −
u
## NC
j
z
j
## (36)
The summation in the constraints is extended to the whole set
of usersJ, but each term is multiplied by the membership
attributionz
j
to ensure no contribution to the shared energy
when the member does not belong to the community. The above
optimization problem has only the shared energy variabless,
hence it is significantly smaller than(35)and this is exploited in
the subsequent decomposition.
C. Procedure for SP Decomposition
The Separation Problem(32)can be formulated by exploiting
the membership variables as the following MILP
min
z
## 
i∈I
## Δ
## M
i
z
i
## −v(z)
s.t.1≤
## 
i∈I
z
i
## ≤|I|−1
z∈{0,1}
## |I|
## (37)
The objective function involves an inner maximization prob-
lem so that a min-max structure seems to appear. Sincev(z)
compares with minus sign, the problem is actually a standard
minimization problem.
To further increase the efficiency of the algorithm, since the
computation ofv
## W/O
(z)involves significantly less variables
thanv
## W
(z), the restriction of(37)without the aggregator
fixingz
## A
=0is solved first. If the optimal value is enough
to identify a coalition to add toΓ, then it is added without
fully solving(37). Otherwise, also the case with the aggregator
## (z
## A
=1) is analyzed. The following steps summarize the above
procedure:
1) solve(37)with the additional constraintz
## A
=0to get the
optimal valueω
## S
## 0
and the corresponding optimal solution
## K
## S
## 0
## ;
2) addK
## S
## 0
toΓin the first stage ifω
## M
## −ω
## S
## 0
## ≥ε,inthe
second if
## ˆ
θ
## LC
## −ω
## S
## 0
## ≥ε;
3) otherwise,solve(37)withtheadditionalconstraintz
## A
## =1
to getω
## S
## .
Moreover, to further speed-up calculations, the separation
problem generates a constraint whenever an incumbent in-
teger solution violates the least core value of the master
problem.
It is worth noting that the adoption of more complex EC
models has marginal impact on the proposed approach, as long
as changes to the objective function and constraints are mixed-
integer convex, which is a de-facto standard in energy modelling.
In such a case, the mathematical framework, the decomposition
technique and also the developed open-sourceEnergyCommu-
nity.jl[43]andTheoryOfGames.jl[44]tools proposed in this
study can be easily adapted.
## VII. C
## ASESTUDY
## A. Description
To validate the methodology, we applied the proposed ap-
proach to a realistic case study that describes ECs of various
sizes (10–100) for a peri-urban area in Italy. Yet, the approach
does not depend on specifics of the Italian case. The demand data
have been adapted from the dataset measured from a Portuguese
substation[50], whose consumption patterns are similar to the
Italian ones, with average peak demand in the range 12–40 kW.
Given their abundance, solar and wind resources have been
considered, and their time series have been obtained from[51].
To avoid market distortion, the market prices of 2019 have been
selected.
## B. Users Composition
In this study, we considered EC with sizes of 10, 20, 30, 50
and 100 members, which aligns to expected values in the Italian
context. To stress the computational performances, about 70%
of the members are prosumers that may install PV, wind and/or
battery technologies with variable asset availability and costs;
users n. 5, 7 and 10 are pure consumers with no assets. To keep
results comparable and highlight trends, the ECs with size larger
than 10 have been obtained by replicating the composition of the
10-user EC. For instance, in the 30-user EC, members 11 and 21
perfectly match user 1. This is justified by the observation that
typical consumers in the power grid do have similar habits and,
consequently, similar demand patterns. However, the methodol-
ogy is absolutely general and applicable to ECs of any size and
composition.
C. Main Techno-Economic Parameters
The cost of installing photovoltaic (PV) systems is between
1.4 and 1.7€/kWp, with a space limitation up to 100 kWp. Wind
turbines cost 3 k€/kW. Lithium batteries cost 400€/kWh plus
200€/kW (converter) and have a round-trip efficiency of 92%.
ThelifetimeofPVis25years,whilewindturbines,batteries,and
converters have a lifespan of 20, 15, and 10 years, respectively.
Yearly maintenance charges have been estimated between 1
and 2% of the initial investment. Purchase and selling prices
of 18 c€/kWh, including taxes, and 5 c€/kWh, respectively,
have been assumed, with monthly peak power charges of 3€
/kW/month[16].
## D. Testing Procedure
To validate the proposed framework, we used Shapley Value,
Nucleolus, Core, Least Core, Variance Core and Variance Least

178IEEE TRANSACTIONS ON ENERGY MARKETS, POLICY, AND REGULATION, VOL. 3, NO. 2, JUNE 2025
Fig. 3.  Comparison of execution time for selected enumerative and iterative methods.
## TABLE I
## P
ERCENTAGEDIFFERENCEBETWEENFINALSURPLUSω
## S
## ANDTRUEVALUE
## TABLE II
## M
## AXIMUMPERCENTAGEDIFFERENCE OFUSERS’ALLOCATIONBETWEEN THE
## ITERATIVE ANDENUMERATIVEAPPROACHES
Fig. 4.  Performances at increasing size of the community of the iterative
techniques - precoalitions[1,|J|].
Core as reward allocation functions for the considered ECs. We
applied the approach described in SectionVand SectionVI
to the latter four allocation functions for all EC configuration.
In order to compare the effectiveness of our approach, we also
performed the complete enumeration of all coalitions. Due to
obvious computational limitations, this has been done only for
the cases of 10 and 20 users. For our approach we performed a
sensitivity analysis on the pre-coalition setΓ, considering up to
6 configurations. The notation of the pre-loading is as follows:
[1]denotes thatΓis pre-loaded with all the coalitions with up to
1 member,[1,|J|]denotes the coalitions with 1 or|J|members,
and so on.
In the following section, we first compare the enumerative
approach with the iterative one, to show the equivalence of their
results but the far superior performances of the latter, which
makes it usable for large ECs. We then perform a sensitivity
analysis with respect to the size of the community, to suggest
guidelines for fair stable reward allocations.
The instances have been solved with the open-sourceEnergy-
Community.jl[43]andTheoryOfGames.jl[44]packages, with
Fig. 5.  Convergence characteristics of the iterative techniques - precoalitions
[1,|J|]; tolerance is 1-100€(about 1–5%).
Fig. 6.  Community surplus (θ
## LC
) by EC size - precoalitions[1,|J|].
underlying MILP solver CPLEX 22.1.0, using 10 threads on a
72-core Xeon computer with 1.2TB RAM. For all sizes below
100 the algorithm stops whenever a relative tolerance of 1% or
an absolute tolerance of 10€is met; for the 100-member EC the
tolerances have been rather set to 5% and 100€.
## VIII. R
## ESULTS
A. Validation of Results
TablesIandIIvalidate the iterative technique described in
SectionsV–VIby reporting the percentage difference between
its results and those of the traditional enumerative computation.

FIORITI et al.: FAIR LEAST CORE: EFFICIENT, STABLE AND UNIQUE GAME-THEORETIC REWARD ALLOCATION IN ENERGY179
Fig. 7.  Benefit (Δ) by user and aggregator: bars represents the average value by user type and error bars highlight the variation.
TableIconfirms that the iterative approaches successfully cap-
ture the true surplus value, computed by complete enumeration,
with differences compatible with the target tolerance. TableII
rather shows the maximum percentage difference across users
in reward allocation. The results show that the VC and VLC
allocations have negligible differences with respect to the exact
solutions, which confirms reproducibility in agreement with the
theory. On the contrary, the computations of allocations in Core
and Least Core are merely feasibility problems. Therefore, it is
natural that different procedures point to allocations that are far
from each other, although having comparable surplus. This is in
agreementwiththetheoryandfurtherconfirmstheimportanceof
finding allocations that are uniquely defined, such as the F[L]C
proposed in SectionIV.
## B. Convergence Characteristics
Figs.3and5highlight the computational time and the con-
vergence characteristics of the proposed method for the 10-
and 20-member ECs. Fig.3clearly confirms that the iterative
algorithm can dramatically reduce computational requirements
by 20×even for the 10-member EC, and beyond 16000×for the
20-member EC. As the enumerative technique required longer
than 2 months to compute, and the computational requirements
grew exponentially, no validation was possible for larger ECs. A
proper pre-loading can have a significant impact on the iterative
algorithm, as the[1,|J|]choice reduced the computational cost
by about 64% with respect to[1]; this is why it is the reference
case for the subsequent investigations.
The efficiency of the algorithm is confirmed in Fig.4, which
shows that the computational cost scales relatively proportional
to the size of the community. This is a significant improve-
ment with respect to the exponential requirements of traditional
techniques illustrated in Fig.3. Moreover, in Fig.5we plot
the differenceω
## M
## −ω
## S
in the first stage and
## ˆ
θ
## LC
## −ω
## S
in
the second stage that is used as convergence criterion of the
proposed algorithm (SectionV); the picture shows that the
algorithm generally converges fairly quickly in a limited number
of iterations.
Overall, these results confirm the potential ability of the
proposed algorithm to scale in size, making game-theoretical
allocation approaches feasible for large ECs. This is helped by
the fact that large ECs are often composed by a small share
of prosumers, which significantly reduces the design decisions
and therefore the expected computational efforts, and that per-
formances of MILP solvers are continuously improving.
C. Benefit and Reward Allocation by Size of Community
Finally, we show in Figs.6and7the effect of EC size
on the community surplus and reward allocation by user,
respectively. Fig.6interestingly shows that the surplus decreases
the larger the EC size till reaching a plateau at 50–100 member.
Indeed, the larger the EC, the lower each user’s market power
within the community, which in turn decreases the LC value.
However, the marginal reduction decreases the larger the com-
munity. As the LC changes, the users’ relative reward alloca-
tion changes, to reflect the different market power within the
community. These results suggest that F[L]C and the proposed
computational effort can account for market power among users
and the aggregator. This allows policy makers to take actions for
limiting possible market distortions.
Fig.7shows the expected user benefit in terms ofΔby mem-
ber type and reward allocation. For simplicity, as for 20-member
EC or larger the users are identically replicated, error bars
depict the maximum and minimum benefit allocation between
thesamemembertypes.First,itisworthnotingthaterrorbarsare
negligible, which means that each member type is remunerated
in the same way. For example, in the 100-members EC there are
10 instances of member types “user1” that are all remunerated
with the same value, which goes in favor of stability and fairness.
These results highlight the role of pure consumers in sup-
porting EC policies and as such policy action should encourage
their participation. On the other hand, depending on the EC
composition, they may obtain undesirable market power that
the proposed algorithm can account for. Therefore, results rec-
ommend policy measures, numerically supported by tools like
the one here proposed, to avoid increasing market share by any
of the players (aggregator, consumers and producers).

180IEEE TRANSACTIONS ON ENERGY MARKETS, POLICY, AND REGULATION, VOL. 3, NO. 2, JUNE 2025
## IX. CONCLUSION
Based on the state-of-the-art on game-theoretic allocations,
this paper proposes and discusses the novel fair stable reward
allocations Fair Core and Fair Least Core for Energy Com-
munities, that are implemented in available, well-engineered
open-source packages. These successfully maximize fairness of
benefit allocation, while enforcing stability by ensuring that no
member is worse off within the community than outside (by
the largest possible margin in the Least Core variant). The new
allocations guarantee uniqueness and reproducibility, which go
in favor of the practical use of the methodology.
Crucially, the work also proposes a row-generation algorithm
to reduce the hitherto staggering computational requirements for
game-theoretic benefit allocations. The new algorithm has been
extensively validated on communities up to 100 members and
can scale even further for very large ECs. The results suggest that
the methodology is a breakthrough that makes game-theoretic
allocations practical for large coalitions while ensuring unique-
ness, reproducibility, and stability. As an example of the man-
agerial and policy insights that the methodology offers, our case
study shows that market power may emerge within members
of the community, which has impact on reward allocation and
hence should be regulated.
This paper lays the foundations for reproducible, fair, and
stable reward allocations, and it can be expected to steer research
in the design of incentive schemes for Energy Communities,
power systems, and beyond. Future studies may explore the
role and remuneration of future flexibility markets in ECs,
including sector-coupled considerations, and/or uncertainties in
major techno-economic parameters.
## R
## EFERENCES
[1] A. Caramizaru and A. Uihlein, “Energy communities: An overview of
energy and social innovation,” European Commission: Joint Research
Centre, Publications Office, Tech. Rep., 2019. [Online]. Available: https:
## //data.europa.eu/doi/10.2760/180576
[2] V. Z. Gjorgievski, S. Cundeva, and G. E. Georghiou, “Social arrange-
ments, technical designs and impacts of energy communities: A review,”
Renewable Energy, vol. 169, pp. 1138–1156, 2021.
[3] J. Lowitzsch, C. E. Hoicka, and F. J. van Tulder, “Renewable energy
communities under the 2019 European clean energy package–Governance
model for the energy clusters of the future?,”Renewable Sustain. Energy
Rev., vol. 122, Apr. 2020, Art. no. 109489.
[4] A. Chis and V. Koivunen, “Coalitional game-based cost optimization of
energy portfolio in smart grid communities,”IEEE Trans Smart Grid,
vol. 10, no. 2, pp. 1960–1970, Mar. 2019.
[5] L.Han,T.Morstyn,andM.McCulloch,“Incentivizingprosumercoalitions
with energy management using cooperative game theory,”IEEE Trans.
Power Syst., vol. 34, no. 1, pp. 303–313, Jan. 2019.
[6] D. Fioriti, G. Lutzemberger, D. Poli, P. Duenas-Martinez, and A. Mi-
cangeli, “Coupling economic multi-objective optimization and multiple
design options: A business-oriented approach to size an off-grid hybrid mi-
crogrid,”Int. J. Electr. Power Energy Syst., vol. 127, 2021, Art. no. 106686.
[7] I. Abada, A. Ehrenmann, and X. Lambin, “On the viability of energy
communities,”Energy J., vol. 34, no. 1, pp. 303–313, Jan. 2020.
[8] N. Rossetto, “Beyond individual active customers: Citizen and renewable
energy communities in the European Union,”IEEE Power Energy Mag.,
vol. 21, no. 4, pp. 36–44, Jul./Aug. 2023.
[9] S. W. Alnaser, S. Z. Althaher, C. Long, Y. Zhou, and J. Wu, “Residential
community with PV and batteries: Reserve provision under grid con-
straints,”J Elec Power Energy Syst., vol. 119, 2020, Art. no. 105856.
[10] A. Basnet and J. Zhong, “Integrating gas energy storage system in a peer-
to-peer community energy market for enhanced operation,”Int. J. Elect.
Power Energy Syst., vol. 118, 2020, Art. no. 105789.
[11] M. Moncecchi, S. Meneghello, and M. Merlo, “A game theoretic approach
for energy sharing in the Italian renewable energy communities,”Appl. Sci.
(Switzerland), vol. 10, no. 22, pp. 1–25, 2020.
[12] N. Patrizi, S. K. LaTouf, E. E. Tsiropoulou, and S. Papavassiliou,
“Prosumer-centric self-sustained smart grid systems,”IEEE Syst. J.,
vol. 16, no. 4, pp. 6042–6053, Dec. 2022.
[13] S. Maleki et al., “The Shapley value for a fair division of group discounts
for coordinating cooling loads,”PLoS One, vol. 15, no. 1, Jan. 2020,
Art. no. e0227049.
[14] N. Kemp, M. S. Siraj, and E. E. Tsiropoulou, “Coalitional demand re-
sponse management in community energy management systems,”Ener-
gies, vol. 16, no. 17, Sep. 2023, Art. no. 6363.
[15] H. T. Nguyen and L. B. Le, “Sharing profit from joint offering of a group
of wind power producers in day ahead markets,”IEEE Trans. Sustain.
Energy, vol. 9, no. 4, pp. 1921–1934, Oct. 2018.
[16] ARERA, “ARERA,” 2024. [Online]. Available: https://www.arera.it/it/
index.htm
[17] U. Amin, J. Hossain, W. Tushar, and K. Mahmud, “Energy trading in local
electricity markets with renewables- A contract theoretic approach,”IEEE
Trans. Ind. Inform., vol. 17, no. 6, pp. 3717–3730, Jun. 2021.
[18] P. Cortés, P. Auladell-León, J. Muñuzuri, and L. Onieva, “Near-optimal
operation of the distributed energy resources in a smart microgrid district,”
J. Cleaner Prod., vol. 252, 2020, Art. no. 119772.
[19] J. Cuenca, E. Jamil, and B. Hayes, “State of the art in energy communities
and sharing economy concepts in the electricity sector,”IEEE Trans. Ind.
Appl., vol. 57, no. 6, pp. 5737–5746, Nov./Dec. 2021.
[20] M. Rayati, M. Bozorg, and R. Cherkaoui, “Coordinating strategic aggrega-
tors in an active distribution network for providing operational flexibility,”
Electric Power Syst. Res., vol. 189, 2020, Art. no. 106737.
[21] L. Wang, W. Gu, Z. Wu, H. Qiu, and G. Pan, “Non-cooperative game-based
multilateral contract transactions in power-heating integrated systems,”
Appl. Energy, vol. 268, no. Oct. 2019, 2020, Art. no. 114930.
[22] C. Feng, F. Wen, S. You, Z. Li, F. Shahnia, and M. Shahidehpour,
“Coalitional game-based transactive energy management in local energy
communities,”IEEE Trans. Power Syst., vol. 35, no. 3, pp. 1729–1740,
## May 2020.
[23] M. Hupez, J.-F. Toubeau, I. Atzeni, Z. De Greve, and F. Vallee, “Pricing
electricity in residential communities using game-theoretical billings,”
IEEE Trans. Smart Grid, vol. 14, no. 2, pp. 1621–1631, Mar. 2023.
[24] S. Cremers, V. Robu, P. Zhang, M. Andoni, S. Norbu, and D. Flynn,
“Efficient methods for approximating the Shapley value for asset sharing
in energy communities,”Appl. Energy, vol. 331, 2023, Art. no. 120328.
[25] M. Tan, Y. Zhou, L. Wang, Y. Su, B. Duan, and R. Wang, “Fair-efficient
energy trading for microgrid cluster in an active distribution network,”
Sustain. Energy, Grids Netw., vol. 26, 2021, Art. no. 100453.
[26] J. Suh and S.-G. Yoon, “Profit-sharing rule for networked microgrids
based on Myerson value in cooperative game,”IEEE Access,vol.9,
pp. 5585–5597, 2021.
[27] Y. Du et al., “A cooperative game approach for coordinating multi-
microgrid operation within distribution systems,”Appl. Energy, vol. 222,
pp. 383–395, 2018.
[28] T. Driessen and Y. Funaki, “Coincidence of and collinearity between game
theoretic solutions,”Operations- Res.-Spektrum, vol. 13, no. 1, pp. 15–30,
## 1991.
[29] R. Van Den Brink and Y. Funaki, “Axiomatizations of a class of equal sur-
plus sharing solutions for Tu-games,”Theory Decis., vol. 67, pp. 303–340,
## 2009.
[30] R. van den Brink, Y. Chun, Y. Funaki, and B. Park, “Consistency, popu-
lation solidarity, and Egalitarian solutions for Tu-games,”Theory Decis.,
vol. 81, pp. 427–447, 2016.
[31] Y. Chun and B. Park, “Population solidarity, population fair-ranking, and
the Egalitarian value,”Int. J. Game Theory, vol. 41, pp. 255–270, 2012.
[32] F. D. Minuto and A. Lanzini, “Energy-sharing mechanisms for energy
community members under different asset ownership schemes and user
demand profiles,”Renewable Sustain. Energy Rev., vol. 168, Oct. 2022,
Art. no. 112859.
[33] D. Fioriti, T. Ferrucci, and D. Poli, “Fairness and reward in energy
communities: Game-theory versus simplified approaches,” inProc. 2023
IEEE Int. Conf. Environ. Elect. Eng. Ind. Commercial Power Syst. Europe,
EEEIC/I CPS Europe, 2023, pp. 1–6.
[34] N. Vespermann, T. Hamacher, and J. Kazempour, “Access economy for
storage in energy communities,”IEEE Trans. Power Syst., vol. 36, no. 3,
pp. 2234–2250, May 2021.
[35] D. Schmeidler, “The nucleolus of a characteristic function game,”SIAM
J. Appl. Math., vol. 17, pp. 1163–1170, 1969.

FIORITI et al.: FAIR LEAST CORE: EFFICIENT, STABLE AND UNIQUE GAME-THEORETIC REWARD ALLOCATION IN ENERGY181
[36] M.Benedek,J.Fliege,andT.-D.Nguyen,“Findingandverifyingthenucle-
olus of cooperative games,”Math. Program., vol. 190, no. 1, pp. 135–170,
## 2021.
[37] F. Belmar, P. Baptista, and D. Neves, “Modelling renewable energy com-
munities: Assessing the impact of different configurations, technologies
and types of participants,”Energy, Sustainability Soc., vol. 13, no. 1, 2023,
Art. no. 18.
[38] Y. Yang, G. Hu, and C. Spanos, “Optimal sharing and fair cost allocation
of community energy storage,”IEEE Trans. Smart Grid, vol. 12, no. 5,
pp. 4185–4194, Sep. 2021.
[39] H. T. Nguyen and L. B. Le, “Bi-objective-Based cost allocation for
cooperative demand-side resource aggregators,”IEEE Trans. Smart Grid,
vol. 9, no. 5, pp. 4220–4235, Sep. 2018.
[40] G. Owen, “On the core of linear production games,”Math. Program.,
vol. 9, no. 1, pp. 358–370, 1975.
[41] N. Öner and G. Kuyzu, “Nucleolus based cost allocation methods for a
class of constrained lane covering games,”Comput. Ind. Eng., vol. 172,
2022, Art. no. 108583.
## [42]
## ̇
I. Muter,  ̧S.
## ̇
I. Birbil, and K. Bülbül, “Simultaneous column-and-row
generation for large-scale linear programs with column-dependent-rows,”
Math. Prog., vol. 142, no. 1, pp. 47–82, Dec. 2013.
[43] D. Fioriti, “EnergyCommunity.jl,” 2024. [Online]. Available: https://
github.com/SPSUnipi/EnergyCommunity.jl
[44] D. Fioriti, “TheoryOfGames.jl,” 2024. [Online]. Available: https://github.
com/SPSUnipi/TheoryOfGames.jl
[45] D. B. Gillies, “Solutions to general non-zero-sum games,” inContributions
to the Theory of Games IV. Annals of Mathematics Studies. Princeton, NJ,
USA: Princeton Univ. Press, 1959, pp. 47–85.
[46] M. Maschler, B. Peleg, and L. S. Shapley, “Geometric properties of the
kernel, nucleolus, and related solution concepts,”Math. Operations Res.,
vol. 4, no. 4, pp. 303–338, 1979.
[47] L.S.Shapley,“Avalueforn-persongames,”inContributionstotheTheory
of Games, Vol. II, Annals of Mathematics Studies. Princeton, NJ, USA:
Princeton Univ. Press, 1953, pp. 307–318.
[48] L. Campagna, G. Rancilio, L. Radaelli, and M. Merlo, “Renewable energy
communities and mitigation of energy poverty: Instruments for policymak-
ers and community managers,”Sustain. Energy, Grids Netw., vol. 39, 2024,
Art. no. 101471.
[49] S. Boyd and L. Vandenberghe,Convex Optimization. Cambridge, U.K.:
## Cambridge Univ. Press, 2004.
[50] A.  Trindade,  “Electricity  load  diagrams  2011-2014  data  set,”
-   [Online].   Available:   https://archive.ics.uci.edu/dataset/321/
electricityloaddiagrams20112014
[51] S. Pfenninger and I. Staffell, “Long-term patterns of European PV output
using 30 years of validated hourly reanalysis and satellite data,”Energy,
vol. 114, pp. 1251–1265, 2016.
Open Access provided by ‘Universitè di Pisa’ within the CRUI CARE Agreement