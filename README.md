# CatFace-Pre

## Features
1. Breed：
有关【花色|种类】概念的设定都是使用 breed 作为键值，但由于 catface 针对**面部**，所以有关 breed 都是针对面部所判断出来的。
<br/>
主要是由于以下的特殊情况：eg. 三花猫面部可能和狸白相似；有的杂合花色面部和身体完全不同。
<br/>
而一般情况下，人类对于花色的分类，依靠的是全身判断，# TODO 实现上可以考虑用全身的图片单独训练一个模型。