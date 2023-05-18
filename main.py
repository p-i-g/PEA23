from manim_slides.slide import Slide
from manim import *
from scipy import optimize


class Intro(Slide):
    def construct(self):
        title = Text("Numerical Modelling")
        self.play(Create(title))
        self.wait(1)
        self.next_slide()

        title.generate_target()
        title.target.shift(0.3 * UP)

        self.play(MoveToTarget(title))
        self.wait(1)

        subtitle = Tex(r"Who? ", "What?", " Why? ", "Where? ", "When? ", "How? ", font_size=32)
        subtitle.shift(0.3 * DOWN)
        self.play(Create(subtitle))
        self.wait(1)
        self.next_slide()
        self.play(FadeOut(title))

        what = Tex("What?", font_size=72)
        self.play(TransformMatchingTex(subtitle, what))
        self.next_slide()


class What(Slide):
    def construct(self):
        what = Tex("What?", font_size=72)
        self.add(what)
        self.play(FadeOut(what))

        chatgpt = Text(
            '"Numerical modeling is a computational technique used to understand and predict the behavior of \n'
            "complex systems. It involves constructing mathematical models, solving them using computational \n" 
            "algorithms, and obtaining numerical solutions. This approach is useful when analytical solutions \n"
            "are difficult or impossible to find. Numerical modeling is applied in various fields to simulate \n"
            'real-world phenomena, analyze different scenarios, and make predictions." - ChatGPT ', font_size=20)

        self.play(Create(chatgpt))
        self.next_slide()
        self.play(FadeOut(chatgpt))


class WhereWhenWhy(Slide):
    def construct(self):
        where_when_why = Tex("Why? ", "Where? ", "When?", font_size=72)
        self.play(Create(where_when_why))

        self.next_slide()

        self.play(FadeOut(where_when_why))

        eqn = MathTex(r'\cos x = x', font_size=32)

        self.play(Create(eqn))

        self.next_slide()

        # eqn.generate_target()
        # eqn.target.shift(6 * UP)

        title = Title(r'$\cos x = x$', include_underline=False)

        ax = Axes(x_range=[0, 1.1, 0.1], y_range=[0, 1., 0.1], tips=False)
        labels = ax.get_axis_labels('x', 'y')

        self.play(Create(ax), Create(labels), Transform(eqn, title))

        cos_plot = ax.plot(np.cos, color=YELLOW)
        x_plot = ax.plot(lambda x: x, color=BLUE)

        cos_label = MathTex(r'y = \cos x', font_size=32).next_to(ax.c2p(0.2, np.cos(0.2)), direction=DOWN)
        x_label = MathTex('y = x', font_size=32).next_to(ax.c2p(0.2, 0.2), direction=UP).shift(0.1 * LEFT)

        self.play(Create(cos_plot), Create(x_plot))
        self.play(Create(cos_label), Create(x_label))

        self.next_slide()

        root = optimize.newton(lambda x: np.cos(x) - x, 1., lambda x: -np.sin(x) - 1)
        dot = Dot(ax.c2p(root, root, 0))

        self.play(Create(dot))

        label = Tex(f'({root:.3f}, {root:.3f})', font_size=24).next_to(dot, buff=0.5)
        self.play(Create(label))
