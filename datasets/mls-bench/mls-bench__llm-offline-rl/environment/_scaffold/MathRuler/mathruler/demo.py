DEMOS = {}

DEMOS["math"] = [
    (
        "Kevin Kangaroo begins hopping on a number line at 0. He wants to get to 1, but he can hop only $\\frac{1}{3}$ of the distance. Each hop tires him out so that he continues to hop $\\frac{1}{3}$ of the remaining distance. How far has he hopped after five hops? Express your answer as a common fraction.",
        "Kevin hops $1/3$ of the remaining distance with every hop. His first hop takes $1/3$ closer. For his second hop, he has $2/3$ left to travel, so he hops forward $(2/3)(1/3)$. For his third hop, he has $(2/3)^2$ left to travel, so he hops forward $(2/3)^2(1/3)$. In general, Kevin hops forward $(2/3)^{k-1}(1/3)$ on his $k$th hop. We want to find how far he has hopped after five hops. This is a finite geometric series with first term $1/3$, common ratio $2/3$, and five terms. Thus, Kevin has hopped $\\frac{\\frac{1}{3}\\left(1-\\left(\\frac{2}{3}\\right)^5\\right)}{1-\\frac{2}{3}} = \\boxed{\\frac{211}{243}}$.",
    ),
    (
        "What is the area of the region defined by the equation $x^2+y^2 - 7 = 4y-14x+3$?",
        "We rewrite the equation as $x^2 + 14x + y^2 - 4y = 10$ and then complete the square, resulting in  $(x+7)^2-49 + (y-2)^2-4=10$, or $(x+7)^2+(y-2)^2=63$. This is the equation of a circle with center $(-7, 2)$ and radius $\\sqrt{63},$ so the area of this region is $\\pi r^2 = \\boxed{63\\pi}$.",
    ),
    (
        "If $x^2+y^2=1$, what is the largest possible value of $|x|+|y|$?",
        "If $(x,y)$ lies on the circle, so does $(x,-y),$ $(-x,-y),$ and $(-x,-y),$ (which all give the same value of $|x| + |y|$), so we can assume that $x \\ge 0$ and $y \\ge 0.$\n\nThen $|x| + |y| = x + y.$  Squaring, we get\n\\[(x + y)^2 = x^2 + 2xy + y^2 = 1 + 2xy.\\]Note that $(x - y)^2 \\ge 0.$  Expanding, we get $x^2 - 2xy + y^2 \\ge 0,$ so $2xy \\le x^2 + y^2 = 1.$  Hence,\n\\[1 + 2xy \\le 2,\\]which means $x + y \\le \\sqrt{2}.$  Equality occurs when $x = y = \\frac{1}{\\sqrt{2}},$ so the maximum value of $|x| + |y|$ is $\\boxed{\\sqrt{2}}.$",
    ),
    (
        "If $f(x)=\\frac{ax+b}{cx+d}, abcd\\not=0$ and $f(f(x))=x$ for all $x$ in the domain of $f$, what is the value of $a+d$?",
        "The condition $f(f(x))$ means that $f$ is the inverse of itself, so its graph is symmetrical about the line $y = x$. With a rational function of this form, we will have two asymptotes: a vertical one at $x=-d/c$ if $cx+d$ does not divide $ax+b$, and a horizontal one at $y=a/c$, if we take the limit of $f(x)$ as $x$ goes to $\\pm\\infty$. In order for $f$ to be its own inverse, the intersection of the asymptotes must lie on the line $y=x$ so that it and its asymptotes reflect onto themselves. This means that $-d/c=a/c$, and therefore $-d=a$ and $a+d=\\boxed{0}$.",
    ),
    (
        "Expand $(2z^2 + 5z - 6)(3z^3 - 2z + 1)$.",
        "$$\\begin{array}{crrrrrrr}\n& & & 3z^3 & & -2z & + 1 & \\\\\n\\times & & & & 2z^2 & +5z & -6 \\\\\n\\cline{1-7}\\rule{0pt}{0.17in}\n& & & -18z^3 & & +12z & -6 & \\\\\n& & +15z^4 & & -10z^2 & +5z & & \\\\\n+ & 6z^5 & & -4z^3 & +2z^2 & & & \\\\\n\\cline{1-7}\\rule{0pt}{0.17in}\n& 6z^5 & +15z^4 & -22z^3 & - 8z^2 &+17z & -6 &\n\\end{array}$$ As such, the answer is $\\boxed{6z^5+15z^4-22z^3-8z^2+17z-6}$.",
    ),
]

DEMOS["gsm8k"] = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is $\\boxed{6}$.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is $\\boxed{5}$.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is $\\boxed{39}$.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is $\\boxed{8}$.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is $\\boxed{9}$.",
    ),
    (
        "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is $\\boxed{29}$.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is $\\boxed{33}$.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is $\\boxed{8}$.",
    ),
]
