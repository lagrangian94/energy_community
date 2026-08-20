

Computing core allocations in cooperative games with an application to
cooperative procurement
## J. Drechsel, A. Kimms
## 
Lehrstuhl f
## ̈
ur Logistik und Operations Research, Mercator School of Management, University of Duisburg–Essen, Lotharstr. 65, 47048 Duisburg, Germany
article info
Article history:
## Received 31 March 2009
## Accepted 16 July 2010
Available online 29 July 2010
## Keywords:
Cooperative game theory
## Core
Mathematical programming
## Procurement
Lot sizing
Inventory games
Supply chain management
abstract
Cooperative game theory defines several concepts for distributing outcome shares in a cooperative
game with transferable utilities. One of the most famous solution concepts is the core which defines a
set of outcome allocations that are stable such that no coalition has an incentive to leave the grand
coalition. In this paper we propose a general procedure to compute a core element (or to detect that no
core allocation exists) which is based on mathematical programming techniques. The procedure
proposed in this paper can be applied to a wide class of cooperative games where the characteristic
function is given by the optimum objective function value of a complex optimization problem. For
cooperative procurement, which is an example from the field of supply chain management where some
literature on the core concept already exists, we prove the applicability and provide computational
results to demonstrate that games with 150 players can be handled.
&2010 Elsevier B.V. All rights reserved.
## 1. Introduction
If several players cooperate, one of the most important
questions is how to distribute the outcome shares. As one
example from the real world, we refer to business networks
(seeLo Nigro and Abbate, in press) where sharing is a topic. Let us
assume that the outcome is a transferable utility such as money,
for instance. For the sake of simplicity we will use the term cost
instead of outcome to have an illustrative wording with the
understanding that players prefer lower outcomes. In other
settings the outcome could be a profit contribution, for instance,
so that players prefer higher outcomes, but it is straightforward to
adapt the following material to such situations.
Cooperative game theory provides several concepts to define
the outcome shares and one concept, the core, which is in
widespread use shall be discussed here: LetNbe the given set of
players and
p
i
be the cost share of playeriAN, which is to be
computed, so that the vector
p¼ðp
## 1
## ,...,p
jNj
Þdenotes a cost
allocation. Furthermore, letc:2
## N
-Rbe the characteristic
function of the game which assigns a cost to each coalition
SDNwhich is the outcome for the coalitionS, if the players inS
cooperate without the players inN\S. To be efficient, one
must have
## X
iAN
p
i
¼cðNÞ:
The cost allocation
pis called an imputation, if
p
i
rcðfigÞfor alli:
This may be considered a desirable property of the cost allocation,
because otherwise a single player has lower cost when acting
alone and cooperating is not rational. Similarly, a coalitionSN,
Sa|, of the players will cooperate with the rest, if
## X
iAS
p
i
rcðSÞð1Þ
holds. An imputation that fulfills the latter condition for all
possible coalitionsSN,Sa|, is called a core element, so
CðN,cÞ¼
pAR
jNj
j
## X
iAN
p
i
¼cðNÞand
## X
iAS
p
i
rcðSÞfor allSN,Sa|
## ()
ð2Þ
is the core. The concept of the core is credited toGillies (1959).
Note that values
p
i
o0 are allowed, butp
i
Z0 automatically
holds for core allocations, if the characteristic functioncis
monotone, i.e.
cðS
## 1
ÞrcðS
## 2
Þfor allS
## 1
## DS
## 2
## DN:
p
i
¼cðNÞ
## X
jAN\fig
p
j
ZcðNÞcðN\figÞZ0:
A cooperative game is said to be subadditive if for any pair of
disjoint player setsS
## 1
## ,S
## 2
DN(withS
## 1
## ,S
## 2
a|andS
## 1
## \S
## 2
## ¼|),
Contents lists available atScienceDirect
journal homepage:www.elsevier.com/locate/ijpe
## Int. J. Production Economics
0925-5273/$-see front matter&2010 Elsevier B.V. All rights reserved.
doi:10.1016/j.ijpe.2010.07.027
## 
Corresponding author.
E-mail addresses:julia.drechsel@uni-due.de (J. Drechsel),
alf.kimms@uni-due.de (A. Kimms).
URL:http://www.msm.uni-due.de/log/ (A. Kimms).
## Int. J. Production Economics 128 (2010) 310–321

we have
cðS
## 1
ÞþcðS
## 2
ÞZcðS
## 1
## [S
## 2
## Þ:
Clearly, there exists an incentive to cooperate in subadditive
games with transferable utilities.
While the definition of the core is well established, the
question of how to determine a core element is open in many
situations. It is well known that for concave games, i.e. games
where
cðS
## 1
[figÞcðS
## 1
ÞZcðS
## 2
[figÞcðS
## 2
## Þ
holds for everyiANandS
## 1
## S
## 2
DN\fig, the core equals the convex
hull of the game’s marginal vectors (seeIchiishi, 1981;Shapley,
1971) and therefore computing a marginal vector yields a core
element. Unfortunately, many games are not concave. Even
worse, for many games it is not clear whether or not the
core is empty. As a result, many contributions to cooperative
games prove the non-emptiness of the core for specific games
(often without computing a core element explicitly, but by
proving that the game is balanced—the Bondareva–Shapley
theorem, seeBondareva, 1963;Shapley, 1967), or prove the
property of concavity for a specific game (or show that the
particular game is not concave by giving a counterexample). If
core elements are explicitly computed tailor-made procedures are
constructed—approaches that are very specific to the underlying
problem and cannot be generalized. We refer to the survey given
byBorm et al. (2001)for examples in this regard. For papers with
structural or general algorithmic results we refer to the following
selection:
Complexity results: The problem of testing core membership
and the problem of proving non-emptiness of the core are
NPcomplete in general (see e.g.Faigle et al., 1997; Fang et al.,
2002;Goemans and Skutella, 2004).
Linear programming and the core:Owen (1975)deals with
linear production games and shows how to obtain a point in
the core by solving a linear program which is an important
contribution, because using the dual variables of an LP
formulation of a problem for defining a core element can be
successful in other applications as well (see e.g.Chen and
Zhang, 2006a). A generalized linear production model is
studied byGranot (1986).Deng et al. (1999)derived important
findings for combinatorial optimization games based on the
central insight that the core is not empty for a particular class
of problems, if and only if an associated linear program has an
integral optimal solution. Several applications to games on
graphs are presented.
Specific applications:Derks and Kuipers (1997)present anOðn
## 2
## Þ
algorithmtocomputeacoreelementinroutinggameswithn
customers.Kuipers (1993)derives several results for information
graph games (a subclass of minimum cost spanning tree games):
withnplayers e.g. the core can be described by at most 2n1
linear constraints. For transportation games,Sa
## ́
nchez-Soriano
(2006)shows that every core element is a particular pairwise
distribution. For assignment games,Sotomayor (2003)reveals
that a unitary core has more than one optimal matching.
Borm et al. (2001)review operations research games including
games related to connection problems (fixed tree, spanning tree),
routing  problems  (Chinese  postman,  travelling  salesman),
scheduling problems (sequencing, permutation, assignment),
production problems (linear production, network flow), and
inventory problems (see Section 3).
Core approximations and core-related concepts:Kamiya
and Talman (1991)propose an algorithm to compute an
approximating core element for balanced games without
side payments by subdividing an appropriate simplex into
smaller simplices.Maschler et al. (1979)discuss geometric
properties of the (
e-) core and related concepts such as
the kernel and the nucleolus. The nucleolus is inside the
core, if the core is not empty (Schmeidler, 1969).Hallefjord
et al. (1995)compute the nucleolus for linear programming
games by means of row generation (an idea that we will
use as well) andG
## ̈
othe-Lundgren et al. (1996)compute the
nucleolus of a vehicle routing game by means of row
generation where the subproblem is a hard-to-solve mixed-
integer programming problem (seeChardaire, 2001, for a note
on the latter paper).Faigle et al. (2001)review the contribu-
tions on computing the nucleolus and provide an algorithm by
their own.
Core properties: A particular selection from the core, the core-
center, is discussed byGonza
## ́
lez-Dı
## ́
az and Sa
## ́
nchez-Rodrı
## ́
guez
(2007).van Velzen et al. (2002)show that if specific sets of
marginal vectors are core elements, then the game is concave.
Hamers et al. (2002)study assignment games to prove that the
extreme points of the core are marginal vectors.Nu
## ́
n
## ̃
ez and
Rafels (1998)prove that the extreme points of the core have a
certain property, the reduced game property.Solymosi (1999)
proves necessary and sufficient conditions under which the
core coincides with the bargaining set for subadditive
games.
Game variants:Faigle (1989)derives core theorems for
cooperative games with restricted cooperation.Bilbao et al.
(2007)define the core for bicooperative games.Okamoto
(2002)investigates properties of the core of a game on a
convex geometry.Lehrer (2002)deals with a temporal aspect
of games where at each stage a budget is distributed among
the players and demonstrates that specific allocation processes
converge to the core. Multiple scenario cooperative games are
such where the cost of a coalition is valued in different
scenarios.Hinojosa et al. (2005)extend the notions of core,
least core, and nucleolus for such settings.
The major contribution of our paper is the proposal of a
general algorithm that can be applied to a broad class of
cooperative games. Section 2.1 describes an algorithm to compute
a core element. This procedure can be modified for variants of the
core, namely the
ecore and the least core, which is discussed
in Section 2.2. Since a core element is not unique in general,
Section 2.3 discusses fairness issues to select a specific core
element. Section 3 applies this algorithm to a specific application,
namely inventory games based on the Wagner–Whitin problem.
In Section 4 a computational study proves that games with 150
players can indeed be attacked. Final conclusions are made in
## Section 5.
- Computing a core element
2.1. A row generation procedure
Formally, the definition of the core (2) specifies a constraint
satisfaction  problem  where  the  number  of  constraints
is exponential with an order of magnitude equal to 2
jNj
.To
tackle such a problem we suggest to use a row generation
procedure.
The master problem of this procedure basically is a relaxed
version of the constraint satisfaction problem given by the core
definition (2). LetSbe a set of coalitions for which the condition
## J. Drechsel, A. Kimms / Int. J. Production Economics 128 (2010) 310–321311

in the core definition should explicitly be stated. Mathematically,
we solve the following linear program:
Master problemMPðSÞ:
minvð3Þ
subject to
## X
iAN
p
i
¼cðNÞ,ð4Þ
## X
iAS
p
i
vrcðSÞ,SAS,ð5Þ
p
i
AR,iAN,ð6Þ
vZ0:ð7Þ
The problem of computing a core element directly corresponds
to the model formulationMPðSÞ, if we chooseS¼2
## N
\f|,Ng, i.e. if
we take 2
jNj
2 constraints of type (5) explicitly into account. Note
that the optimum objective function value of modelMPðSÞ
indicates whether or not the core is empty. If the optimum
solution ofMPðSÞyieldsv¼0, then the problem has a non-empty
core and the values
p
i
in that optimum solution define a cost
allocation which is in the core. On the other hand, if the optimum
solution givesv40, then we know that the cooperative game
instance under consideration has an empty core. It should be
emphasized that the valuesc(S) in the model formulation (3)–(7)
are given parameters which means that in advance we have to
determinec(N) in (4) and thejSjvaluesc(S) in (5), but this
computational burden is not due to our procedure and it is
inherent to the definition of the core which assumes that all
valuesc(S) forSDN,Sa|, are known in advance.
The iterative row generation procedure can be outlined as
follows:
- Define a small initial setS, e.g.S¼ff1g,...,fjNjgg.
- Solve the linear programMPðSÞoptimally.
- Ifv40 then stop. The game instance has an empty core.
- Otherwise,  find  a  coalitionSu=2S(Sua|)  such  that
## P
iASu
p
i
4cðSuÞ. An optimization subproblem
## ^
SPðpÞwhich
searches for a coalitionSuthat violates the core defining
inequality (1) most (
## P
iASu
p
i
cðSuÞis maximized) can, for
instance, be defined.
- If no such coalitionSucan be found (
## ^
SPð
pÞhas a non-positive
optimum objective function value) then stop. The current
values
p
i
define a core allocation.
- Otherwise, updateS¼S[fSug. Return to Step 2.
## 2.2.
eCoreand least-core elements
The proposed procedure computes an element of the core.
Closely related to the core are the concepts of the
ecore and the
least-core. We will now show that slight modifications of our
procedure can handle these core variants as well.
## The
ecore was introduced byShapley and Shubik (1966)and
is defined as
## C
e
ðcÞ¼pAR
jNj
j
## X
iAN
p
i
¼cðNÞand
## X
iAS
p
i
rcðSÞþefor allSN,Sa|
## ()
## ,
whereeARis a given parameter. Apparently, the core is a special
case, i.e.C
## 0
(c)¼C(c). Theecore can be interpreted as follows: If
founding a coalitionSNand quitting the grand coalition incurs a
cost
e40, the grand coalition is stable even if the coalitionSreceives
a cost share larger thanc(S)aslongas
## P
iAS
p
i
rcðSÞþeholds. If for
founding a coalitionSNand quitting the grand coalition a reward
## 
e40 is offered (from a third party), the grand coalition is stable as
long as
## P
iAS
p
i
rcðSÞðeÞis fulfilled. Note that this interpretation
is not meant to imply that the
ecore concept is senseful only if the
grand coalition is given as a starting state.
As stated already, the core of a cooperative game might be
empty. But it should be clear that for a sufficiently large value
e
theecore of the very same game is not empty. On the other
hand, if the core of a game is not empty, then the
ecore lies
within the core for
er0, i.e.C
e
ðcÞDCðcÞ.
Maschler et al. (1979)have formally defined the least-core
## C
## L
(c) as being theecore with smallest possibleeARsuch that
## C
e
ðcÞis not empty, i.e.C
## L
ðcÞ¼
## T
eAR:C
e
ðcÞa|
## C
e
ðcÞ. Hence,Maschler
et al. (1979)describe the least-core as centrally located within the
core, if the core is not empty (
er0), and as a means to reveal the
position of the ‘‘latent’’ core, if the core is empty (
e40).
Our procedure can be adapted to find an element of the
ecore
as follows: Replace (5) in the master problem by
## X
iAS
p
i
vrcðSÞþe,SAS
and replace the subproblem’s objective by
max
## X
iASu
p
i
cðSuÞe:
To compute an element in the least-core, we have to do the
two substitutions that were just described for the
ecore.
Furthermore, we delete the variablevfrom the model so that
the constraints (5) are of the form
## P
iAS
p
i
rcðSÞþe. In addition,e
must be a real-valued decision variable (and not a parameter in
the master problem) and the objective function of the master
problem must be
min
e
instead of (3). The valueeto be used in the subproblem equals the
objective function value of the most recent master problem.
In the following, we will confine our discussion to the core and
we will not treat its related concepts explicitly.
## 2.3. Fairness
In this subsection we assume that the game has a non-empty
core which can be checked by running the algorithm introduced so
far. In general, however, the core does not consist of a single element
only. So we may want to find a core cost allocation with a certain
characteristic. While every core element is stable in the sense that
no coalition has an incentive to leave the grand coalition, an
arbitrary core element may not be considered as being fair. This
aspect can be taken into account by using a slightly different master
problem formulation. Two variants shall be suggested here.
VariantI: The absolute cost shares should deviate as little as
possible among the players:
Master problemMP
## I
ðSÞ:
min
## PP
subject to
## X
iAN
p
i
¼cðNÞ,
## X
iAS
p
i
rcðSÞ,SAS,
PZp
i
,iAN,
## Prp
i
,iAN,
p
i
AR,iAN,
## P,PAR:
Note that the use of this master problem formulation yields an
equal distribution of the cost shares, i.e.
p
i
## ¼
cðNÞ
jNj
## J. Drechsel, A. Kimms / Int. J. Production Economics 128 (2010) 310–321312

if and only if
jSj
jNj
r
cðSÞ
cðNÞ
for allSDN
is true. To prove this, letSbe an arbitrary coalition:
cðSÞZ
## X
iAS
p
i
## ¼
## X
iAS
cðNÞ
jNj
## ¼
jSj
jNj
cðNÞ:
Geometrically,MP
## I
ðSÞdetermines the core element which is
closest according to the Euclidean metric to the midpoint of the
polyhedronf
pZ0j
## P
iAN
p
i
¼cðNÞg(seeFig. 2for an illustration).
Note that this midpoint might be outside the core which means
that the core element is not equal to the nucleolus in general
where the nucleolus is the centroid of the core constraints
(Straffin, 1993).
VariantII: The percentage cost savings should deviate as little
as possible among the players:
Master problemMP
## II
ðSÞ:
min
## PP
subject to
## X
iAN
p
i
¼cðNÞ,
## X
iAS
p
i
rcðSÞ,SAS,
PZp
i
=cðfigÞ,iAN,
## Prp
i
=cðfigÞ,iAN,
p
i
AR,iAN,
## P,PAR:
It is easy to verify that a distribution of the cost shares
proportional to thec({i}) values, i.e.
p
i
## ¼
cðfigÞ
## P
jAN
cðfjgÞ
cðNÞ,
defines a core element if and only if
## P
iAS
cðfigÞ
## P
iAN
cðfigÞ
r
cðSÞ
cðNÞ
for allSDN
holds. To prove this, letSbe an arbitrary coalition:
cðSÞZ
## X
iAS
p
i
## ¼
## X
iAS
cðfigÞ
## P
jAN
cðfjgÞ
cðNÞ¼
## P
iAS
cðfigÞ
## P
jAN
cðfjgÞ
cðNÞ:
From  a  geometric  perspective,MP
## II
ðSÞcomputes  the  core
element which is closest according to the Euclidean metric to a
specific convex combination of thejNjcorner points of the
polyhedronf
pZ0j
## P
iAN
p
i
¼cðNÞandp
i
rcðfigÞfor alliANgwhere
o
i
¼cðfigÞ=
## P
jAN
cðfjgÞis the weight of the corner pointk
i
with
p
i
rcðfigÞbeing unnecessary to define that particular cornerk
i
(a numerical example will be seen in Section 3.2).
- An application: inventory games
3.1. Cooperative procurement
Due to the growing number of supply chain management
success stories companies are more and more involved nowadays.
Business units are forced to think in terms of complex supply
chain networks rather than in terms of isolated decision making.
Arshinder and Deshmukh (2008)discuss coordination issues in
supply chains. Figuring out if and how cooperation with others
can improve own performance is an established paradigm today.
Game theory methods are adequate to investigate such situations.
We refer toLeng and Parlar (2005),Meca and Timmer (2008), and
Nagarajan and Sos
## ̆
is
## ́
(to appear)for a recent survey. In this section
we will study cooperation in placing orders.Tella and Virolainen
(2005)examine motives behind joint ordering. Not only private
companies consider joint ordering as an important topic, but
public bodies usually join cooperative procurement programs to
form a purchasing alliance and to benefit from economies of scale.
Note that we use the terms procurement problem, ordering
problem, and lot sizing problem as synonyms in this paper.
EOQ/EPQ-related games:Meca et al. (2004)term the situation
of cooperative procurement an inventory game. They use the
well-known economic order quantity (EOQ) model as a basis.
The advantage of their approach is that an analytical treatment
of the problem is possible and that for all results a closed form
solution (a formula) can be derived. The results obtained have
been extended for a game based on the economic production
quantity (EPQ) model with shortages byMeca et al. (2003).
This work has been extended even further byMeca (2007)for
generalized holding costs. Temporary price discounts are
included byMeca et al. (2007).Dror and Hartman (2007)
discuss a multi-product extension of the EOQ-based game.
Newsvendor-related games:Parlar (1988)was probably the
first who considered games based on random demand
inventory problems.Hartman et al. (2000)andSlikker et al.
(2001)use the newsvendor inventory model as a basis for a
cooperative game that is called inventory centralization game
in the literature.Hartman et al. (2000)reveal assumptions
under which newsvendor based games do not have an empty
core. They also provide conditions which are equivalent to the
condition of the Bondareva–Shapley theorem under the
assumption of independent and identically distributed (iid)
demands.M
## ̈
uller et al. (2002)prove that the core of the
newsvendor game is nonempty regardless of the joint
distribution of the random demands.Hartman and Dror
(2003)present a greedy optimization procedure to compute
a solution that minimizes the cost of centralization. Conditions
on the holding and penalty costs that ensure subadditivity of
the game are derived byHartman and Dror (2005). A dynamic
version of the newsvendor game is presented byDror et al.
(2008). The possibility of transshipment between the news-
vendors is added to the problem bySlikker et al. (2005).O
## ̈
zen
et al. (2007)prove the non-emptiness of the core in a
newsvendor setting where goods are delivered to warehouses
before they reach the retailers. Demands are realized when the
goods reach the warehouse and order quantities can be
reallocated before shipping from the warehouses to the
retailers takes place.Chen and Zhang (2006b)show that
determining whether an allocation is in the core of a news-
vendor game isNPhard even in very simple settings. Price-
dependent demand and quantity discounts are introduced by
Chen (2007)(see alsoGuardiola et al., 2007c, for quantity
discounts in a different setting) and demand updates are
discussed byO
## ̈
zen et al. (in press).
Wagner–Whitin-related games: Inventory games with discrete
dynamic demand over multiple periods have been introduced
byGuardiola et al. (in press)and are called production
inventory games. The costs taken into account are holding
costs, backlogging costs, and production costs. It is shown that
the core is not empty and a point in the core can be computed
in polynomial time. An axiomatic foundation for the point
computed is provided byGuardiola et al. (2008). Setup costs
are added byGuardiola et al. (2006)who coined the name
setup inventory games.Chen and Zhang (2006a)compute a
core element of a setup inventory game in polynomial time
by  utilizing  linear  programming  duality  (an  approach
which heavily relies on a specific model formulation while
our approach is more general and independent from the
## J. Drechsel, A. Kimms / Int. J. Production Economics 128 (2010) 310–321313

mathematical properties of a specific model formulation).
Another paper on setup inventory games is the one byvan den
Heuvel et al. (2007a)where the well-establishedWagner–
Whitin (1958)model is used as a starting point. So do we:
A single decision maker has to make order decisions for a
single item. Given a planning horizon ofTtime periods, a demand
d
t
has to be met without backlogging and without shortages in
every period. In order to meet the demand in periodtone may
place an order intor before. If the order is placed before, the
ordered items must be stored in inventory. If a fixed costs
t
is
incurred whenever an order is placed in periodtand a unit
holding costh
t
is charged for every item on stock at the end of
periodt, then we face the classical trade-off between saving fixed
costs versus saving holding costs—a lot sizing problem occurs. In
addition to that a unit ordering costp
t
is incurred for each item
being ordered. The decision to be made isq
t
, the quantity to be
ordered in periodt. Depending on the order quantities and the
demand we haveI
t
units of the item on stock at the end of a
periodt. This problem can be couched as a mixed-integer
programming problem using the following notation:
## Parameters:
Tthe number of periods
## I
## 0
the quantity on stock at the beginning of period 1,
w.l.o.g.I
## 0
## ¼0
d
t
the demand in periodt
s
t
the fixed cost for placing an order in periodt
h
t
the holding cost coefficient for having one unit on stock
at the end of periodt
p
t
the unit cost of ordering an item in periodt
Ma big number, e.g.M¼
## P
t
d
t
Decision variables:
q
t
the quantity to be ordered in periodt
## I
t
the number of items on stock at the end of periodt
x
t
¼1, if an order is placed in periodt(0, otherwise)
Given this notation the problem of finding order quantities
which minimize cost can be stated as follows:
min
## X
## T
t¼1
ðs
t
x
t
þh
t
## I
t
þp
t
q
t
Þð8Þ
subject toI
t
## ¼I
t1
þq
t
## d
t
,t¼1,...,T,ð9Þ
q
t
rMx
t
,t¼1,...,T,ð10Þ
q
t
## ,I
t
Z0,t¼1,...,T,ð11Þ
x
t
Af0,1g,t¼1,...,T:ð12Þ
The objective (8) is to minimize the sum of fixed and quantity
dependent ordering costs and holding costs. The inventory balance (9)
states that at the end of a period we have on stock what was there at
the beginning of the period plus what was ordered in that period
minus period demand. If the order quantity is positive in periodtthen
the indicator variablex
t
must be set to one as stated by (10). The
domain of the decision variables is specified in (11) and (12). Note
that due toI
t
Z0 all demand must be fulfilled right in time.
Although it is well-known that the Wagner–Whitin problem
(8)–(12) can be solved very efficiently (seeFedergruen and Tzur,
1991; Wagelmans et al., 1992;Aggarwal and Park, 1993), the
problem remains to be an optimization problem where no closed
formula can be provided to specify the result.
Thesituationdescribeduptohereassumesasingleplayer.Now
assume that multiple players consider to cooperate and let the set of
all players beNwithjNjZ2. A cooperation here means that these
players decide to place orders together. More formally this means
that each playeriANfaces a demandd
it
to be met in periodt.Ifthe
players from the setNcooperate, they face a joint demand
d
t
ðNÞ¼
## X
iAN
d
it
## :
Followingvan den Heuvel et al. (2007a)the problem to be
solved for this group of players is the following:
cðNÞ¼min
## X
## T
t¼1
ðs
t
x
t
þh
t
## I
t
þp
t
q
t
Þð13Þ
subject to
## I
t
## ¼I
t1
þq
t
## d
t
ðNÞ,t¼1,...,T,ð14Þ
q
t
rMx
t
,t¼1,...,T,ð15Þ
q
t
## ,I
t
Z0,t¼1,...,T,ð16Þ
x
t
Af0,1g,t¼1,...,T,ð17Þ
wherec(N) is the total cost for ordering jointly. Note thatcðSÞZ0,
for allSDN, andcð|Þ¼0 by construction.
The property of subadditivity holds which is a consequence of
the totally balanced character of this type of games (Guardiola
et al., 2006). Hence, the defined cooperative procurement game
gives a reason to the players in the setNto form the grand
coalition. This, however, is true only if the characteristic functionc
measures a transferable utility (TU). Since we consider cost
(money) here, this is the case and the procurement game
obviously is a TU game. The characteristic functioncis monotone,
i.e.cðS
## 1
ÞrcðS
## 2
Þfor allS
## 1
## DS
## 2
DN, because ofd
t
ðS
## 1
## Þrd
t
ðS
## 2
## Þ.
van den Heuvel et al. (2007a)prove that the cooperative game
(N,c), which is based on model (13)–(17), has a non-empty core.
However, they provide no way to compute a core element for the
general case. Nevertheless,van den Heuvel et al. (2007a)show that
this game is not concave in general. For two special cases they can
prove concavity, namely the two-period caseT¼2 and the case where
all players face equal demand, i.e.d
it
## ¼d
jt
for alli,jANandt¼1,y,T.
Our procedure can be applied now (a numerical example will
be given in the subsequent subsection) where the subproblem
## ^
SPð
pÞis defined to be the following:
## Subproblem
## ^
SPð
pÞ:
maxobjþ
## X
iAN
p
i
z
i
ð18Þ
subject toI
t
## ¼I
t1
þq
t
## 
## X
iAN
d
it
z
i
,t¼1,...,T,ð19Þ
q
t
rMx
t
,t¼1,...,T,ð20Þ
obj¼
## X
## T
t¼1
ðs
t
x
t
þh
t
## I
t
þp
t
q
t
Þ,ð21Þ
q
t
## ,I
t
Z0,t¼1,...,T,ð22Þ
x
t
Af0,1g,t¼1,...,T,ð23Þ
z
i
Af0,1g,iAN,ð24Þ
objZ0:ð25Þ
Note that a coalitionSuto be considered in the master problem
is found, if the optimum objective function value of the
## J. Drechsel, A. Kimms / Int. J. Production Economics 128 (2010) 310–321314

subproblem is positive.Suis defined by the values of thez
i
variables (z
i
¼1 indicatesiASu) andcðSuÞ¼obj, i.e.objdenotes the
objective  function  value  of  the  Wagner–Whitin  problem
which corresponds to the coalitionSu. The symbols
p
i
denote
parameter values. These values are computed by solving the
master problem (
p
i
was a decision variable in the master
problem).van den Heuvel et al. (2007b)prove that the
subproblem isNPcomplete.
3.2. A numerical example
A small example shall now be given to illustrate the working
principle of our procedure. We use here Example 5 fromvan den
Heuvel et al. (2007a). (Note that this example contains a small
mistake in the original paper: misprinted parameter values.
Van den Heuvel sent to us the corrected values upon request.)
This example consists ofjNj¼3 players andT¼6 periods of
time. The parameter values are defined to be as given in
Table 1,I
## 0
## ¼0.
Table 2reveals the full details of this example game, but it
should be emphasized that most of these details need not to be
known in advance and are computed if needed during the course
of our procedure. The core of that game can be illustrated
graphically (cp.Maschler et al., 1979)asinFig. 1.
Iteration1: Solving Wagner–Whitin problems optimally, we
computec({1,2,3})¼1393,c({1})¼644,c({2})¼511, andc({3})¼
- Then we solveMP({{1},{2},{3}}) optimally:
minv
subject to
p
## 1
þp
## 2
þp
## 3
## ¼1393,
p
## 1
## vr644,
p
## 2
## vr511,
p
## 3
## vr483,
p
## 1
## ,p
## 2
## ,p
## 3
## AR,
vZ0:
The optimum solution isv¼0 and
p¼ð644,511,238Þ:
Now we solve the subproblem. The result isz
## 1
## ¼1,z
## 2
¼1, and
z
## 3
¼0 which is true, becausep
## 1
þp
## 2
¼115541029¼cðf1,2gÞ.
Iteration2: SolveMP({{1},{2},{3},{1,2}}) optimally which
means to solve the master problem from iteration 1 with the
additional constraint
p
## 1
þp
## 2
## vr1029:
The optimum solution isv¼0 and
p¼ð518,511,364Þ:
The subproblem is called which gives the resultz
## 1
## ¼0,z
## 2
## ¼1,
andz
## 3
¼1, becausep
## 2
þp
## 3
¼8754869¼cðf2,3gÞ.
Iteration3: SolveMP({{1},{2},{3},{1,2},{2,3}}) optimally which
means to solve the master problem from iteration 2 with the
additional constraint
p
## 2
þp
## 3
## vr869:
The optimum solution isv¼0 and
p¼ð524,505,364Þ:
Calling the subproblem reveals that this solution is in the core
and the algorithm terminates. To check that this is true, we could
manually test if
p
## 1
þp
## 3
rcðf1,3gÞholds. One finds 524þ364¼
888o1004, so the algorithm worked correct.
It should be remarked that
p¼ð524,505,364Þis an extreme
point of the core. Whether or not an extreme point (or a vertex
point or an inner point) comes out depends on the solution
procedure for the master problem, a linear program.
The core of every cooperative TU game is a closed and bounded
convex set (seeOwen, 1995). For our specific example, for
instance, the core can be specified as follows:
CðN,cÞ¼f
l
## 1
ð524,505,364Þþl
## 2
ð524,389,480Þ
þ
l
## 3
ð640,389,364Þjl
## 1
þl
## 2
þl
## 3
¼1 andl
## 1
## ,l
## 2
## ,l
## 3
## Z0g:
## Since
## P
iAN
p
i
¼cðNÞthe core is anðjNj1Þdimensional polyhe-
dron (cf.Fig. 1). By the way, for this example the least-core is the
ecore withe¼38:67:
## C
## L
ðcÞ¼fð562:67,427:67,402:67Þg:
If we replaceMPðSÞbyMP
## I
ðSÞand apply the described
procedure  to  the  example  again,  the  cost  allocation
p¼ð524,434:5,434:5Þcomes out.Fig. 2illustrates the situation.
If we replaceMPðSÞbyMP
## II
ðSÞinstead and apply the procedure
described to the example, the cost allocation
p¼ð547:68,
434:57,410:76Þcomes out.Fig. 3provides an illustration. The
corner points mentioned in Section 2.3 are
k
## 1
¼ð399,511,483Þ,
k
## 2
¼ð644,266,483Þ,  andk
## 3
¼ð644,511,238Þwith  weights
o
## 1
## ¼0:39,o
## 2
¼0:31, ando
## 3
## ¼0:29.
## 3.3. Generality
Up to here we have used the classical Wagner–Whitin problem
to illustrate our ideas. But, it should be emphasized that our
approach is much more general and that other problems can be
attacked straightforwardly. A few variants should be mentioned:
Player-dependent cost coefficients: For the sake of simplicity,
we have assumed up to here that all players use the same cost
coefficients. If we assume that this is not true, let us say that
order unit costs and holding costs are player-dependent (e.g.,
because each player has to pay transportation costs and runs
its own warehouse, respectively), we have to take into account
thatp
it
andh
it
values must be used.
Warehouse capacity constraints: If the players run a common
warehouse, the size of the warehouse may be limited. We refer
toLove (1973)for an early work on this kind of problem, but
where cooperative games are not dealt with. Note that other
capacity restrictions may be added to the Wagner–Whitin
problem as well (see, e.g.,Rosling, 1993).
Multi-level structures: If each player has implemented a multi-
stage order processing (e.g. central warehouses, regional
## Table 1
Parameter values for the example.
t12 3 456
d
## 1t
## 1551492011
d
## 2t
## 111117314
d
## 3t
## 208111119
s
t
## 01001327177111
h
t
## 5 3 542 1
p
t
## 111 478 8
## Table 2
A cooperative game with three players.
## S
## |
## {1}{2}{3}{1,2}{1,3}{2,3}{1,2,3}
c(S)0644511483102910048691393
## J. Drechsel, A. Kimms / Int. J. Production Economics 128 (2010) 310–321315

warehouses and so on), the players may cooperate on each
stage. For the sake of simplicity, imagine the case of a serial
structure with a unique downstream successor of each stage.
For an early work on serial lot sizing problems, we refer to
Zangwill (1969)(a paper which considers solving the lot sizing
problem without cooperation).
To the best of our knowledge, these variants have not been
studied as cooperative games, so it is an open question whether or
not the core of these games is empty. Note, our procedure can be
applied to all these problems simply by modifying the model
formulation of the subproblem. Details are out of the scope of this
paper and left to the reader.
It is remarkable to note that our procedure is not confined to
variants of the Wagner–Whitin problem. The idea is valid for
every game where the characteristic function is evaluated by
solving an optimization problem (the subproblem might be a
linear but also a non-linear mixed-integer model). Note, different
model types for the subproblem will require different solution
procedures, but the proposed idea does not rely on a specific
algorithm for the subproblem. This, of course, does not mean that
the run-time performance is equally good for all games (finding a
core element isNPhard in general).
Finally, note that the task of solving the subproblem might be
thorny in general. But it should be remarked that this is not due to
our approach, but due to the problem itself. Even in the definition
(2) of the core thec(S) values are needed, so there is no way to
avoid computing them. In contrast to the general definition of the
core, our procedure usually requires to compute just a few, but
not allc(S) values which is an advantage.
If complex optimization problems are studied, the subproblem
probably has to be solved by heuristics (at least if problem
instances of practical size should be attacked). Note that in
our application already the subproblem is hard to solve
theoretically, but standard software succeeds in spite of this.
It is not a problem that computingc(S) may be hard, because
the definition (2) of the core does not require optimum solutions.
The valuec(S) can be seen as the ‘‘best’’ result that can be
reached by a coalitionSwhich for hard problems means that
the ‘‘best’’ result is a heuristic result. In such cases, we simply
have to implement a heuristic to solve the subproblem during
the course of our routine. So even hard optimization problems
do not contradict our suggested approach. Anyhow, one has to
be careful when applying heuristics, because it may happen
that a particular game is subadditive when the subproblem
is solved optimally to compute thec(S) values, but the property
of subadditivity does not hold when thec(S) values are
heuristic values (i.e. upper bounds, if the underlying problem
is a minimization problem). If one can prove that subadditivity
holds with regard to optimumc(S) values, one could use
a pragmatic approach when solving the subproblem heuristically:
whenever a collection of non-empty, disjoint player setsS
## 1
,y,S
n
will be better off when acting alone, i.e.cðS
## 1
ÞþþcðS
n
## Þo
cð
## S
n
k¼1
## S
k
Þ(recall thatcðÞare heuristic values now), we
could simply replacecð
## S
n
k¼1
## S
k
Þin the right hand side of (5)
withcðS
## 1
ÞþþcðS
n
Þand restart the procedure of computing
a core element again with the modified master problem.
What happened can simply be seen as that someone found a
new (better) heuristic solution for the problem with the players
## S
n
k¼1
## S
k
## .
Fig. 1.The corefðp
## 1
## ,p
## 2
## ,p
## 3
Þgof the example.
Fig. 2.Illustration ofMP
## I
ðSÞ.
Fig. 3.Illustration ofMP
## II
ðSÞ.
## J. Drechsel, A. Kimms / Int. J. Production Economics 128 (2010) 310–321316

- Computational study
To test the proposed procedure we have implemented it using
the commercial software AMPL/CPLEX version 10.0.0. All tests
were conducted on Intel Celeron hardware with 2GHz. The key
parameter for cooperative planning isjNj, the number of players,
since the definition of the core and so the full master problem
formulation grows in the order of 2
jNj
## .
We discovered that the run-time performance to find core cost
allocations is very promising. Even the largest instances in our
test-beds were solved within reasonable time (e.g., forjNjr20 we
observed a run-time of a few minutes). The run-time per iteration
usually was fractions of a second (jNjr30 andT¼6). Instead of
reporting CPU-times we prefer to provide information on the
number of iterations required, because this measure does not
depend on the hardware used and on the efficiency of the
software implementation.
Besides testing different instance characteristics, we also
studied the impact of the problem-specific part of our procedure,
i.e. the subproblem, on the performance. Therefore, we tested
several alternative subproblem formulations. Recall that
## ^
SPð
pÞ
searches for coalitions violating (1) most. As a variant, we
implemented subproblems looking for (1)-violating coalitions of
smallest size (SPuð
pÞ=SP
## 00
ðpÞ) and of largest size (
## ~
SPuðpÞ=
## ~
## SP
## 00
ðpÞ),
respectively. The difference betweenSPuð
pÞandSP
## 00
ðpÞ(
## ~
SPuðpÞand
## ~
## SP
## 00
ðpÞ) is that in every iterationSP
## 00
ðpÞ(
## ~
## SP
## 00
ðpÞ) looks for a
coalition size greater (smaller) than the size found in the previous
iteration where the size counter is reset if no (1)-violating
coalition is found so that no coalition is overlooked.
4.1. Test-bed 1: basic instances
To start with, we generated random instances withT¼6 and
parameter values being random integers which were drawn from
the following intervals with uniform distribution:d
it
## A½0,20,
s
t
## A½0,200,h
t
A½0,10, andp
t
A½0,15. The number of players was
varied systematically:
jNjAf5,10,15,20,25,30g:
For each valuejNjwe generated 15 instances. To keep the
computational burden within reasonable limits, we terminated
the procedure after 20000 iterations in case that a core element
has not been detected up to this point.
Table 3shows the absolute number of iterations. We provide
the average number of iterations over 15 instances (#iter) as well
as the minimum and maximum number, respectively.Table 3also
reveals the efficiency of our approach. While the original core
definition (2) requires 2
jNj
2 constraints of type (5), on average
our procedure used justjNjconstraints (from Step 1) plus
(#iter1) generated ones. The values provided inTable 3are
## Table 3
Test-bed 1—average number of iterations/average percentage of required constraints (min/max number of iterations).
T¼6SPuð
pÞSP
## 00
ðpÞ
## ~
SPuð
pÞ
## ~
## SP
## 00
ðpÞ
## ^
SPð
pÞ
jNj¼5MPðSÞ7.9/39.78%6.8/36.00%5.6/32.00%5.7/32.44%4.0/26.67%
## (4/12)(4/8)(4/8)(4/8)(4/4)
## MP
## I
ðSÞ
## 4.2/27.33%4.1/26.89%3.2/24.00%3.2/24.00%2.6/22.00%
## (2/10)(2/8)(2/8)(2/8)(2/4)
## MP
## II
ðSÞ
## 1.1/17.11%1.1/17.11%1.1/17.11%1.1/17.11%1.1/17.11%
## (1/2)(1/2)(1/2)(1/2)(1/2)
jNj¼10MPðSÞ139.7/14.55%117.9/12.41%17.5/2.60%19.1/2.75%10.5/1.91%
## (25/371)(29/222)(9/47)(9/50)(9/17)
## MP
## I
ðSÞ
## 29.9/3.80%23.7/3.20%7.4/1.60%7.5/1.61%7.3/1.59%
## (11/61)(10/38)(3/20)(3/18)(3/10)
## MP
## II
ðSÞ
## 3.5/1.23%3.5/1.22%1.9/1.07%1.9/1.07%1.9/1.06%
## (1/9)(1/9)(1/4)(1/4)(1/3)
jNj¼15MPðSÞ1159.1/3.58%545.1/1.71%24.3/0.12%23.0/0.11%15.3/0.09%
## (27/3975)(33/1436)(14/88)(14/92)(14/28)
## MP
## I
ðSÞ
## 246.2/0.79%155.4/0.52%16.9/0.09%15.1/0.09%20.8/0.11%
## (56/407)(48/268)(7/92)(7/67)(9/33)
## MP
## II
ðSÞ
## 44.3/0.18%33.3/0.14%5.7/0.06%5.7/0.06%7.7/0.07%
## (10/112)(9/74)(3/8)(3/8)(3/13)
jNj¼20MPðSÞ
## 2
## 44461:1=40:43%
## 1394.7/0.13%315.9/0.03%179.6/0.02%61.8/0.01%
ð19=420000Þ(20/7782)(19/2416)(19/1220)(19/519)
## MP
## I
ðSÞ
## 982.9/0.10%518.3/0.05%65.5/0.01%52.1/0.01%56.7/0.01%
## (374/1828)(218/907)(11/425)(11/374)(17/228)
## MP
## II
ðSÞ
## 203.7/0.02%139.5/0.02%8.7/0.00%8.7/0.00%15.2/0.00%
## (9/628)(8/424)(4/12)(4/12)(5/40)
jNj¼25MPðSÞ
## 6
## 49063:8=0:03%
## 2
## 47455:1=0:02%
## 355.9/0.00%298.7/0.00%29.5/0.00%
ð24=420000Þð24=420000Þ(24/1489)(24/1317)(24/75)
## MP
## I
ðSÞ
## 2647.0/0.01%1263.7/0.00%575.6/0.00%221.2/0.00%131.4/0.00%
## (1638/6295)(840/1734)(17/4278)(17/1291)(36/484)
## MP
## II
ðSÞ
## 597.7/0.00%342.5/0.00%77.6/0.00%41.1/0.00%30.0/0.00%
## (187/1231)(122/750)(8/630)(8/281)(12/63)
jNj¼30MPðSÞ
## 5
## 47886:3=0:00%
## 2
## 46960:6=0:00%
## 1022.8/0.00%463.1/0.00%50.0/0.00%
ð35=420000Þð31=420000Þ(29/11954)(29/4343)(29/169)
## MP
## I
ðSÞ
## 6452.4/0.00%2477.1/0.00%420.3/0.00%210.9/0.00%192.8/0.00%
## (3573/14247)(1661/3536)(20/2818)(20/1485)(56/455)
## MP
## II
ðSÞ
## 1696.9/0.00%975.5/0.00%37.9/0.00%31.5/0.00%65.7/0.00%
## (308/2819)(226/1443)(11/302)(11/188)(42/84)
## J. Drechsel, A. Kimms / Int. J. Production Economics 128 (2010) 310–321317

defined to be
ðjNjþiter1Þ100
## 2
jNj
## 2
which is the percentage of constraints that were actually used.
We examine all variants of the master problem formulation
(MPðSÞ,MP
## I
ðSÞ,MP
## II
ðSÞ) in combination with all variants of the
subproblem formulation (SPuð
pÞ,SP
## 00
ðpÞ,
## ~
SPuðpÞ,
## ~
## SP
## 00
ðpÞ,
## ^
SPðpÞ).
Table 3reveals that the combination ofMPðSÞwithSPuð
pÞfor
jNj¼25 gave the following result in our test with 15 instances: six
instances reached the iteration limit of 20000 and were
terminated before finding a core element. The average number
of iterations over all 15 instances is larger than 9063.8 (the six
instances which were aborted before finding a core element
contribute a value of 20000 each to this average number). Our
efficiency measure reveals that this corresponds to an average of
just 0.03% of generated constraints. At least one of the 15
instances terminated after 24 iterations, while the maximum
number of iterations required is larger than 20000 (in six cases).
When looking at the results, we observe that the procedure is
indeed efficient, because only a small fraction of constraints is
really generated (note that a percentage value of 0.00% in the
table means that the percentage value being measured is smaller
than 0.005%). As expected, the number of iterations grows faster
than linear with a growing number of players, although the
efficiency increases with an increasing number of players. For
jNj¼30, for instance, we know that 2
## 30
## 2410
## 9
constraints exist,
but for the combinationMP
## II
ðSÞ=
## ^
SPðpÞnot more than just about
100 of those were generated on average. It turns out that the
master problemMP
## II
ðSÞhas a better performance than the master
problemMP
## I
ðSÞwhich in turn is better thanMPðSÞno matter what
subproblem formulation is employed. Regarding the subproblems
it becomes clear that searching for small coalitions (SPuð
pÞand
## SP
## 00
ðpÞ) performs much worse than its alternatives which is
plausible, because adding constraints which affect only a few
players tends to enforce smaller solution changes from iteration
to iteration than adding constraints which affect many players.
For a large number of players, the subproblem
## ~
## SP
## 00
ðpÞoutperforms
## ~
SPuð
pÞ. The subproblem
## ^
SPðpÞseems to be equally good to
subproblem
## ~
## SP
## 00
ðpÞ(no clear dominance) in terms of average
values when master problemMP
## II
ðSÞis used. For the worst case,
i.e. regarding the maximum number of required iterations,
## ^
SPð
pÞ
seems to be the better choice.
4.2. Test-bed 2: grand coalition size dependent fixed costs
The instances in the first test-bed have fixed costss
t
## A½0,200
which does not depend on the number of players under
consideration. Hence, for a growing number of players the
Wagner–Whitin problems for large coalitions tend to have an
optimum solution which equals a lot-for-lot ordering policy, i.e.
## Table 4
Test-bed 2—average number of iterations/average percentage of required constraints (min/max number of iterations).
T¼6SPuð
pÞSP
## 00
ðpÞ
## ~
SPuð
pÞ
## ~
## SP
## 00
ðpÞ
## ^
SPð
pÞ
jNj¼5MPðSÞ7.9/39.78%6.8/36.00%5.6/32.00%5.7/32.44%4.0/26.67%
## (4/12)(4/8)(4/8)(4/8)(4/4)
## MP
## I
ðSÞ
## 4.2/27.33%4.1/26.89%3.2/24.00%3.2/24.00%2.6/22.00%
## (2/10)(2/8)(2/8)(2/8)(2/4)
## MP
## II
ðSÞ
## 1.1/17.11%1.1/17.11%1.1/17.11%1.1/17.11%1.1/17.11%
## (1/2)(1/2)(1/2)(1/2)(1/2)
jNj¼10MPðSÞ192.8/19.75%123.2/12.94%40.3/4.82%29.3/3.75%11.1/1.96%
## (25/511)(21/288)(9/128)(9/76)(9/31)
## MP
## I
ðSÞ
## 8.8/1.74%9.1/1.77%5.5/1.42%5.3/1.40%3.5/1.23%
## (1/20)(1/24)(1/22)(1/19)(1/7)
## MP
## II
ðSÞ
## 1.3/1.00%1.3/1.00%1.2/1.00%1.2/1.00%1.2/1.00%
## (1/4)(1/4)(1/3)(1/3)(1/3)
jNj¼15MPðSÞ2124.0/6.53%877.4/2.72%331.5/1.05%142.7/0.48%22.5/0.11%
## (66/9289)(37/2118)(16/1507)(16/423)(14/89)
## MP
## I
ðSÞ
## 47.6/0.19%36.0/0.15%11.5/0.08%11.2/0.08%7.6/0.07%
## (7/115)(7/77)(3/49)(3/44)(3/12)
## MP
## II
ðSÞ
## 3.8/0.05%3.9/0.05%2.1/0.05%2.1/0.05%1.9/0.05%
## (1/16)(1/14)(1/4)(1/4)(1/3)
jNj¼20MPðSÞ
## 6
## 410630:9=1:02%
## 1
## 46124:2=0:59%
## 1637.7/0.16%651.5/0.06%106.6/0.01%
ð27=420000Þð45=420000Þ(19/10996)(19/2502)(19/728)
## MP
## I
ðSÞ
## 166.7/0.02%101.9/0.01%79.1/0.01%32.1/0.00%15.0/0.00%
## (6/683)(6/333)(2/1020)(2/339)(2/86)
## MP
## II
ðSÞ
## 14.1/0.00%13.1/0.00%3.4/0.00%3.5/0.00%3.3/0.00%
## (1/70)(1/72)(1/6)(1/6)(1/7)
jNj¼25MPðSÞ
## 12
## 416097:6=0:05%
## 9
## 414814:2=0:04%
## 2
## 44090:2=0:01%
## 2142.4/0.01%153.3/0.00%
ð24=420000Þð74=420000Þð24=420000Þ(24/8636)(24/752)
## MP
## I
ðSÞ
## 183.0/0.00%128.8/0.00%19.8/0.00%17.7/0.00%11.1/0.00%
## (28/605)(26/379)(4/190)(4/159)(4/26)
## MP
## II
ðSÞ
## 20.6/0.00%16.1/0.00%3.2/0.00%3.2/0.00%3.6/0.00%
## (1/103)(1/93)(1/7)(1/7)(1/8)
jNj¼30MPðSÞ
## 14
## 418676:3=0:00%
## 8
## 415421:9=0:00%
## 4552.7/0.00%3024.5/0.00%120.3/0.00%
ð144=420000Þð148=420000Þ(49/17076)(42/16420)(29/564)
## MP
## I
ðSÞ
## 384.5/0.00%244.5/0.00%10.3/0.00%10.4/0.00%15.8/0.00%
## (2/1440)(2/676)(2/37)(2/40)(2/38)
## MP
## II
ðSÞ
## 49.7/0.00%34.5/0.00%4.0/0.00%4.0/0.00%4.3/0.00%
## (1/290)(1/164)(1/9)(1/9)(1/10)
## J. Drechsel, A. Kimms / Int. J. Production Economics 128 (2010) 310–321318

the order quantity in a period equals demand in that period,
because fixed costs are relatively low. In our second test-bed we
therefore use instances withs
t
A½100ðjNj=51Þ,100ðjNj=5þ1Þ
so that even for a large number of players the Wagner–Whitin
problems remain to be non-trivial. All other parameter values
were chosen as in test-bed 1. Especially the number of instances
per parameter constellation is 15 again.
Basically, we can make the same observations as with the first
test-bed. Again, it turns out that the combinationsMP
## II
ðSÞ=
## ~
## SP
## 00
ðpÞ
andMP
## II
ðSÞ=
## ^
SPðpÞare the best performers (seeTable 4). It is
interesting to note that the procedure terminates faster when
compared with the results from Test-Bed 1 forMP
## II
ðSÞ=
## ~
## SP
## 00
ðpÞand
## MP
## II
ðSÞ=
## ^
SPðpÞ. ForjNj¼30 and the combinationMP
## II
ðSÞ=
## ^
SPðpÞ, for
instance, less than 35 constraints out of the 2
## 30
## 2ð410
## 9
## Þare
used on average. The reason seems to be that different coalitions
have much more different costs now so that the constraints which
are really needed to define the core can much more easily be
detected as a look atFig. 1may illustrate.
4.3. Test-bed 3: very large procurement problems
To prove that the proposed approach can be used even if the
underlying problem is large, we tested instances of different size
measured in terms ofT. We have used
TAf6,12,18,24,30,36,42,48,54,100g:
The number of players was set tojNj¼25. All other parameters
were chosen as in test-bed 1. Again, 15 random instances were
used for each parameter setting.Table 5provides the results for
the combinationsMP
## I
ðSÞ=
## ^
SPðpÞandMP
## II
ðSÞ=
## ^
SPðpÞ.
It is remarkable to observe that the proposed procedure tends
to terminate after fewer iterations the larger the optimization
problem is. This is not strictly true, as we can see, but the trend is
obvious. A reason for this is not clear to us.
4.4. Test-bed 4: very large number of players
Eventually, we examined the impact of the numberjNjof
players on the performance. We used
jNjAf30,50,100,150g
and the parameter settings as in test-bed 1. Once more, 15
random instances per parameter constellation were solved. The
procedure is halted if 10000 iterations are reached. The results for
the combinationsMP
## I
ðSÞ=
## ^
SPðpÞandMP
## II
ðSÞ=
## ^
SPðpÞare provided in
## Table 6.
We see that the number of iterations grows, if the number of
players does, but still very large instances can be solved with
reasonable computational effort. The results are even more
impressive, if we bring to our mind again that forjNj¼150
players 2
## 150
2, which is more than 10
## 45
, core defining inequal-
ities exist.
## 5. Conclusion
In situations where multiple players decide to cooperate,
which is the case in supply chains, for example, the question
of how to distribute the outcome shares plays a dominant role.
If the objective of the underlying problem is optimization
of  a  transferable  utility  (such  as  minimizing  costs,  for
instance), cooperative game theory defines the concept of
the  core.  A  core  element  specifies  an  allocation  of  the
objective function value among the players such that the grand
coalition is stable, i.e. no smaller coalition has an incentive to
work alone.
While the definition of the core is clear, the question of how to
determine a core element is open in many situations. It is well
known that for concave games, the core can be described by the
game’s marginal vectors. Unfortunately, many games are not
concave. Even worse, for many games it is not clear whether or
not the core is empty.
Thus, a procedure is proposed which can be applied to very
general settings, especially settings where the underlying pro-
blem of the game is a complex optimization problem without a
closed form solution. The proposed procedure is an iterative row
generation procedure. Only slight modifications are necessary to
apply our procedure if instead of the core the
ecore or the least-
core should be investigated.
Our procedure can further be refined. We suggest not to
compute an arbitrary core element, but a core element that can be
considered as fair. Two variants are studied by us, minimizing the
deviation of the absolute cost shares and minimizing the
deviation of the percentage cost improvements. In a computa-
tional study it turns out that computing a fair core element can be
done even more efficient than computing an arbitrary core
element where the latter variant is the most efficient one.
By means of a specific application, namely the well-established
Wagner–Whitin problem, we demonstrate the working principle
of this procedure. We use a cooperative procurement game which
is known from the literature. This game is subadditive. Our
mathematical programming procedure computes a core element,
## Table 5
Test-bed 3—average number of iterations/average percentage of required
constraints (min/max number of iterations).
jNj¼25
## MP
## I
ðSÞ=
## ^
SPðpÞMP
## II
ðSÞ=
## ^
SPðpÞ
## T¼6131.4/0.00%30.0/0.00%
## (36/484)(12/63)
## T¼12145.7/0.00%18.3/0.00%
## (39/1027)(3/34)
## T¼1889.1/0.00%14.5/0.00%
## (43/356)(5/46)
## T¼24124.5/0.00%10.2/0.00%
## (21/612)(2/30)
## T¼3068.3/0.00%5.9/0.00%
## (19/140)(3/16)
## T¼3670.4/0.00%7.3/0.00%
## (28/201)(1/22)
## T¼4246.6/0.00%5.0/0.00%
## (12/116)(1/21)
## T¼4840.6/0.00%4.8/0.00%
## (19/57)(2/14)
## T¼5441.1/0.00%3.1/0.00%
## (17/85)(1/8)
## T¼10017.7/0.00%1.5/0.00%
## (6/50)(1/3)
## Table 6
Test-bed 4—average number of iterations/average percentage of required
constraints (min/max number of iterations).
## T¼6
## MP
## I
ðSÞ=
## ^
SPðpÞMP
## II
ðSÞ=
## ^
SPðpÞ
jNj¼30192.8/0.00%65.7/0.00%
## (56/455)(42/84)
jNj¼50501.3/0.00%270.2/0.00%
## (140/2555)(109/1074)
jNj¼1002763.8/0.00%1071.1/0.00%
## (772/7083)(689/1351)
jNj¼150
## 1
## 43751:0=0:00%
## 1
## 43122:7=0:00%
ð1735=410000Þð1280=410000Þ
## J. Drechsel, A. Kimms / Int. J. Production Economics 128 (2010) 310–321319

which can be done efficiently even if the number of players is
large as shown by a computational study.
Future work should apply this procedure to other applications.
A few extensions of the considered procurement problem are
already suggested in the paper. But other applications can be
attacked in the same manner.
## Acknowledgment
We thank Wilco van den Heuvel for providing us the corrected
example data.
## References
Aggarwal, A., Park, J.K., 1993. Improved algorithms for economic lot-size problems.
## Operations Research 14, 549–571.
Arshinder, K.A., Deshmukh, S.G., 2008. Supply chain coordination: perspective,
empirical studies and research directions. International Journal of Production
## Economics 115, 316–335.
Bilbao, J.M., Ferna
## ́
ndez, J.R., Jime
## ́
nez, N., Lo
## ́
pez, J.J., 2007. The core and the weber
set for bicooperative games. International Journal of Game Theory 36,
## 209–222.
Bondareva, O.N., 1963. Some applications of linear programming methods to the
theory of cooperative games. Problemy Kibernetiki 10, 119–133.
Borm, P., Hamers, H., Hendrickx, R., 2001. Operations research games: a survey.
TOP—An Official Journal of the Spanish Society of Statistics and Operations
## Research 9, 139–216.
Chardaire, P., 2001. The core and nucleolus of games: a note on a paper by G
## ̈
othe–
Lundgren et al. Mathematical Programming, Series A 90, 147–151.
Chen, X., 2007. Inventory centralization games with price-dependent demand and
quantity discount. Working Paper, University of Illinois at Urbana-Champaign.
Chen, X., Zhang, J., 2006a. Duality approaches to economic lot-sizing games.
Working Paper, University of Illinois at Urbana-Champaign.
Chen, X., Zhang, J., 2006b. A stochastic programming duality approach to inventory
centralization games. Working Paper, University of Illinois at Urbana-
## Champaign.
Deng, X., Ibaraki, T., Nagamochi, H., 1999. Algorithmic aspects of the core of
combinatorial optimization games. Mathematics of Operations Research 24,
## 751–766.
Derks, J., Kuipers, J., 1997. On the core of routing games. International Journal of
## Game Theory 26, 193–205.
Dror, M., Guardiola, L.A., Meca, A., Puerto, J., 2008. Dynamic realization games in
newsvendor inventory centralization. International Journal of Game Theory
## 37, 139–153.
Dror, M., Hartman, B.C., 2007. Shipment consolidation: who pays for it and how
much? Management Science 53 78–87.
Faigle, U., 1989. Cores of games with restricted cooperation. ZOR—Models and
Methods of Operations Research 33, 405–422.
## Faigle, U., Kern, W., Fekete, S., Hochst
## ̈
attler, W., 1997. On the complexity of testing
membership in the core of min cost spanning tree games. International Journal
of Game Theory 26, 361–366.
Faigle, U., Kern, W., Kuipers, J., 2001. On the computation of the nucleolus of a
cooperative game. International Journal of Game Theory 30, 79–98.
Fang, Q., Zhu, S., Cai, M., Deng, X., 2002. On computational complexity of
membership test in flow games and linear production games. International
Journal of Game Theory 31, 39–45.
Federgruen, A., Tzur, M., 1991. A simple forward algorithm to solve general
dynamic lot sizing models withnperiods inO(nlogn)orO(n) time.
## Management Science 37, 909–925.
Gillies, D.B., 1959. Solutions to general non-zero-sum games. In: Tucker, A.W.,
Luce, R.D. (Eds.), Contributions to the Theory of Games IV.  Princeton
University Press, Princeton, pp. 47–85.
Goemans, M.X., Skutella, M., 2004. Cooperative facility location games. Journal of
## Algorithms 50, 194–214.
## G
## ̈
othe-Lundgren, M., J
## ̈
ornsten, K., V
## ̈
arbrand, P., 1996. On the nucleolus of the basic
vehicle routing game. Mathematical Programming 72, 83–100.
## Gonza
## ́
lez-Dı
## ́
az, J., Sa
## ́
nchez-Rodrı
## ́
guez, E., 2007. A natural selection from the core of
a TU game: the core-center. International Journal of Game Theory 36, 27–46.
Granot, D., 1986. A generalized linear production model: a unified model.
## Mathematical Programming 34, 212–222.
Guardiola, L.A., Meca, A., Puerto, J., 2006. Coordination in periodic review inventory
situations. Working Paper, Universidad Miguel Herna
## ́
ndez de Elche.
Guardiola, L.A., Meca, A., Puerto, J., in press. Production-inventory games: a new
class of totally balanced combinatorial optimization games. Games and
Economic Behavior, doi:10.1016/j.geb.2007.02.003.
Guardiola, L.A., Meca, A., Puerto, J., 2008. Production–Inventory games and PMAS
games: characterization of the owen point. Mathematical Social Sciences 56,
## 96–108.
Guardiola, L.A., Meca, A., Timmer, J., 2007c. Cooperation and profit allocation in
distribution chains. Decision Support Systems 44, 17–27.
## Hallefjord, A., Helmig, R., J
## ̈
ornsten, K., 1995. Computing the nucleolus when the
characteristic function is given implicitly: a constraint generation approach.
International Journal of Game Theory 24, 357–372.
Hamers, H., Klijn, F., Solymosi, T., Tijs, S., Villar, J.P., 2002. Assignment games
satisfy the CoMa-property. Games and Economic Behavior 38, 231–239.
Hartman, B.C., Dror, M., 2003. Optimizing centralized inventory operations in a
cooperative game theory setting. IIE Transactions 35, 243–257.
Hartman, B.C., Dror, M., 2005. Allocation of gains from inventory centralization in
newsvendor environments. IIE Transactions 37, 93–107.
Hartman, B.C., Dror, M., Shaked, M., 2000. Cores of inventory centralization games.
Games and Economic Behavior 31, 26–49.
Hinojosa, M.A., Ma
## ́
rmol, A.M., Thomas, L.C., 2005. Core, least core and nucleolus for
multiple scenario cooperative games. European Journal of Operational
## Research 164, 225–238.
Ichiishi, T., 1981. Super–modularity: applications to convex games and the greedy
algorithm for LP. Journal of Economic Theory 25, 283–286.
Kamiya, K., Talman, D., 1991. Simplicial algorithm for computing a core element
in a balanced game. Journal of the Operations Research Society of Japan 34,
## 222–228.
Kuipers, J., 1993. On the core of information graph games. International Journal of
## Game Theory 21, 339–350.
Lehrer, E., 2002. Allocation processes in cooperative games. International Journal
of Game Theory 31, 341–351.
Leng, M., Parlar, M., 2005. Game theory applications in supply chain management:
a review. INFOR 43, 187–220.
Lo Nigro, G., Abbate, L., in press. Risk assessment and profit sharing in business
networks, International Journal of Production Economics, doi:10.1016/j.ijpe.
## 2009.08.014.
Love, S.F., 1973. Bounded production and inventory models with piecewise
concave costs. Management Science 20, 313–318.
Maschler, M., Peleg, B., Shapley, L.S., 1979. Geometric properties of the kernel,
nucleolus, and related solution concepts. Mathematics of Operations Research
## 4, 303–338.
Meca, A., 2007. A core–allocation family for generalized holding cost games.
Mathematical Methods of Operations Research 65, 499–517.
## Meca, A., Garcı
## ́
a-Jurado, I., Borm, P., 2003. Cooperation and competition in
inventory games. Mathematical Methods of Operations Research 57, 481–493.
Meca, A., Guardiola, L.A., Toledo, A., 2007.p-Additive games: a class of totally
balanced games arising from inventory situations with temporary discounts.
TOP—An Official Journal of the Spanish Society of Statistics and Operations
## Research 15, 322–340.
Meca, A., Timmer, J., 2008. Supply chain collaboration. In: Kordic, V. (Ed.), Supply Chain
Theory and Applications. I-Tech Education and Publishing, Vienna, pp. 1–18.
## Meca, A., Timmer, J., Garcı
## ́
a-Jurado, I., Borm, P., 2004. Inventory games. European
Journal of Operational Research 156, 127–139.
## M
## ̈
uller, A., Scarsini, M., Shaked, M., 2002. The newsvendor game has a nonempty
core. Games and Economic Behavior 38, 118–126.
## Nagarajan, M., Sos
## ̆
is
## ́
, G., to appear. Game-theoretic analysis of cooperation among
supply chain agents: review and extensions. European Journal of Operational
Research. Available online:/http://ssrn.com/abstract=900744S.
## Nu
## ́
n
## ̃
ez, M., Rafels, C., 1998. On extreme points of the core and reduced games.
Annals of Operations Research 84, 121–133.
## O
## ̈
zen, U., Fransoo, J., Norde, H., Slikker, M., 2008. Cooperation between multiple
newsvendors with warehouses. Manufacturing & Service Operations Manage-
ment 10, 311–324.
## O
## ̈
zen, U., Sos
## ̆
ic
## ́
, G., Slikker, M., 2007. A collaborative decentralized distribution system
with demand updates. Working Paper, Technische Universiteit Eindhoven.
Okamoto, Y., 2002. Some properties of the core on convex geometries.
Mathematical Methods of Operations Research 56, 377–386.
Owen, G., 1975. On the core of linear production games. Mathematical
## Programming 9, 358–370.
Owen, G., 1995. Game Theory, third ed. Academic Press, San Diego.
Parlar, M., 1988. Game theoretic analysis of the substitutable product inventory
problem with random demands. Naval Research Logistics 35, 397–409.
Rosling, K., 1993. A capacitated single-item lot-size model. International Journal of
## Production Economics 30–31, 213–219.
## Sa
## ́
nchez-Soriano, J., 2006. Pairwise solutions and the core of transportation
situations. European Journal of Operational Research 175, 101–110.
Schmeidler, D., 1969. The nucleolus of a characteristic function game. SIAM Journal
of Applied Mathematics 17, 1163–1170.
Shapley, L.S., 1967. On balanced sets and cores. Naval Research Logistics Quarterly
## 14, 453–460.
Shapley, L.S., 1971. Cores of convex games. International Journal of Game Theory 1,
## 11–26.
Shapley, L.S., Shubik, M., 1966. Quasi-cores in a monetary economy with
nonconvex preferences. Econometrica 34, 805–827.
Slikker, M., Fransoo, J., Wouters, M., 2001. Joint ordering in multiple news–vendor
problems: a game–theoretical approach. Working Paper, Technische Universi-
teit Eindhoven.
Slikker, M., Fransoo, J., Wouters, M., 2005. Cooperation between multiple news-
vendors with transshipments. European Journal of Operational Research 167,
## 370–380.
Solymosi, T., 1999. On the bargaining set, kernel and core of superadditive games.
International Journal of Game Theory 28, 229–240.
Sotomayor, M., 2003. Some further remark on the core structure of the assignment
game. Mathematical Social Sciences 46, 261–265.
## J. Drechsel, A. Kimms / Int. J. Production Economics 128 (2010) 310–321320

Straffin, P.D., 1993. Game Theory and Strategy. The Mathematical Association of
America, Washington, DC.
Tella, E., Virolainen, V.-M., 2005. Motives behind purchasing consortia. Interna-
tional Journal of Production Economics 93–94, 161–168.
van den Heuvel, W., Borm, P., Hamers, H., 2007a. Economic lot sizing games.
European Journal of Operational Research 176, 1117–1130.
van den Heuvel, W., Kundakcioglu, O.E., Geunes, J., Romeijn, H.E., Sharkey, T.C.,
Wagelmans, A.P.M., 2007b. Integrated market selection and production
planning: complexity and solution approaches. Working Paper, Erasmus
## University Rotterdam.
van Velzen, B., Hamers, H., Norde, H., 2002. Convexity and marginal vectors.
International Journal of Game Theory 31, 323–330.
Wagelmans, A., van Hoesel, S., Kolen, A., 1992. Economic lot sizing: anO(nlogn)
algorithm that runs in linear time in the Wagner–Whitin case. Operations
Research 40, S145–S156.
Wagner, H.M., Whitin, T.M., 1958. Dynamic version of the economic lot size model.
## Management Science 5, 89–96.
Zangwill, W.I., 1969. A backlogging model and a multi-echelon model of a dynamic
economic lot size production system—a network approach. Management
## Science 15, 506–527.
## J. Drechsel, A. Kimms / Int. J. Production Economics 128 (2010) 310–321321