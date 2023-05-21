import numpy as np
from manim_slides.slide import Slide
from manim import *
from scipy import optimize
from scipy import interpolate as interp


class Intro(Slide):
    def construct(self):
        title = Text("Numerical Modelling")
        self.play(Create(title))
        self.wait(0.1)
        self.next_slide()

        title.generate_target()
        title.target.shift(0.3 * UP)

        self.play(MoveToTarget(title))

        subtitle = Tex(r"Who? ", "What?", " Why? ", "Where? ", "When? ", "How? ", font_size=32)
        subtitle.shift(0.3 * DOWN)
        self.play(Create(subtitle))
        self.wait(0.1)
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
        self.wait(0.1)
        self.next_slide()
        self.play(FadeOut(chatgpt))


class WhereWhenWhy(Slide):
    def construct(self):
        where_when_why = Tex("Why? ", "Where? ", "When?", font_size=72)
        self.play(Create(where_when_why))

        self.next_slide()

        self.play(FadeOut(where_when_why))

        eqn = MathTex(r'\cos x = x', font_size=72)

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

        self.next_slide()

        self.play(*[FadeOut(m) for m in self.mobjects])

        how = Tex("How?", font_size=72)
        self.play(Create(how))

        # self.next_slide()
        # self.play(FadeOut(how))
        #


class BisectionMethod(Slide):
    def construct(self):
        bisection_method = Tex('Bisection Method (Binary Search)', font_size=72)
        self.play(Create(bisection_method))
        self.wait(0.1)
        self.next_slide()
        title = Title('Bisection Method (Binary Search)', include_underline=False)
        self.play(Transform(bisection_method, title))

        main_eqn = MathTex(r'\cos x', r'=', r'x')
        self.play(Create(main_eqn))

        self.next_slide()

        # morph the equation
        zero_eqn = MathTex(r'\cos x', r'-', 'x', r'=', r'0')
        self.play(TransformMatchingTex(main_eqn, zero_eqn))

        self.next_slide()

        ax = Axes(x_range=[0, 1.1], y_range=[-0.5, 1.2])
        graph = ax.plot(lambda x: np.cos(x) - x, color=BLUE)

        zero_eqn.generate_target()
        zero_eqn.target.scale(0.75).shift(2 * UP + 5 * RIGHT)

        self.play(Create(ax), MoveToTarget(zero_eqn))
        self.play(Create(graph))

        x_start = 0.
        x_end = 1.

        start_dot = Dot(ax.c2p(x_start, 0), color=ORANGE)
        end_dot = Dot(ax.c2p(x_end, 0), color=ORANGE)
        line = Line(color=ORANGE, start=ax.c2p(x_start, 0), end=ax.c2p(x_end, 0), buff=0)
        self.next_slide()
        self.play(Create(start_dot), Create(line), Create(end_dot))

        for j in range(10):
            x_mid = (x_end + x_start) / 2

            if np.cos(x_mid) - x_mid > 0:
                x_start = x_mid
            else:
                x_end = x_mid

            if j < 4:
                vert_line = DashedLine(start=ax.c2p(x_mid, 0), end=ax.c2p(x_mid, np.cos(x_mid) - x_mid))
                dot = Dot(ax.c2p(x_mid, np.cos(x_mid) - x_mid), color=(RED if np.cos(x_mid) - x_mid > 0 else BLUE))
                self.next_slide()
                self.play(Create(vert_line))
                self.play(Create(dot))
                self.next_slide()

                self.play(FadeOut(vert_line), FadeOut(dot))

            line.generate_target()
            line.target.put_start_and_end_on(ax.c2p(x_start, 0), ax.c2p(x_end, 0))
            start_dot.generate_target()
            start_dot.target.set_x(ax.c2p(x_start)[0])
            end_dot.generate_target()
            end_dot.target.set_x(ax.c2p(x_end)[0])
            self.play(MoveToTarget(line), MoveToTarget(start_dot), MoveToTarget(end_dot))

        self.next_slide()


class Derivatives(Slide):
    def construct(self):
        newtons_method = Tex("Newton-Rhapson Method", font_size=72)
        self.play(Create(newtons_method))
        self.wait(0.1)
        self.next_slide()
        self.play(FadeOut(newtons_method))

        calc = Tex("Before that: calculus in 1 minute", font_size=72)
        self.play(Create(calc))
        self.wait(0.1)
        self.next_slide()
        self.play(FadeOut(calc))

        derivatives = Tex("Derivatives", font_size=72)
        self.play(Create(derivatives))

        title = Title("Derivatives", include_underline=False)
        self.play(Transform(derivatives, title))
        self.next_slide()

        v_def = MathTex(r'v = \frac{\Delta x}{\Delta t}')
        self.play(Create(v_def))
        self.next_slide()

        v_def.generate_target()
        v_def.target.shift(2. * UP + 1.5 * LEFT)

        ax = Axes(x_range=[0, 1, 2], y_range=[0, 1, 2], tips=False)
        labels = ax.get_axis_labels('t', 'x(t)')
        self.play(Create(ax), MoveToTarget(v_def), Create(labels))

        self.next_slide()

        x_square = ax.plot(lambda x: x * x * x, color=BLUE)
        self.play(Create(x_square))

        self.next_slide()

        delta = ValueTracker(0.2)

        dot1 = Dot([0, 0, 0], color=YELLOW)
        dot2 = Dot([0, 0, 0], color=YELLOW)

        secant = Line(start=LEFT, end=RIGHT, color=GREEN)

        text = DecimalNumber(0, num_decimal_places=3, color=YELLOW, font_size=32).shift(1.5 * RIGHT + 2 * DOWN) \
            .next_to(ax.c2p(0.7, 0.7 ** 3), buff=0.5)

        def dot1_updater(m: Mobject):
            m.set_x(ax.c2p(0.7 - delta.get_value(), (0.7 - delta.get_value()) ** 3)[0])
            m.set_y(ax.c2p(0.7 - delta.get_value(), (0.7 - delta.get_value()) ** 3)[1])

        def dot2_updater(m: Mobject):
            m.set_x(ax.c2p(0.7 + delta.get_value(), (0.7 + delta.get_value()) ** 3)[0])
            m.set_y(ax.c2p(0.7 + delta.get_value(), (0.7 + delta.get_value()) ** 3)[1])

        def line_updater(m: Mobject):
            m.put_start_and_end_on(ax.c2p(0.7 - delta.get_value(), (0.7 - delta.get_value()) ** 3),
                                   ax.c2p(0.7 + delta.get_value(), (0.7 + delta.get_value()) ** 3))

        def text_updater(m: Mobject):
            m.set_value(1.47 + delta.get_value() ** 2)

        dot1.add_updater(dot1_updater)
        dot2.add_updater(dot2_updater)
        secant.add_updater(line_updater)
        text.add_updater(text_updater)

        self.add(dot1, dot2, secant, text)

        self.play(delta.animate.set_value(0.))

        [m.clear_updaters() for m in self.mobjects]

        self.next_slide()
        self.play(*[FadeOut(m) for m in self.mobjects])


class Integrals(Slide):
    def construct(self):
        integrals = Tex("Integrals", font_size=72)
        self.play(Create(integrals))

        title = Title("Integrals", include_underline=False)
        self.play(Transform(integrals, title))
        self.next_slide()

        v_def = MathTex(r'v = \frac{\Delta x}{\Delta t}')
        v_def_2 = MathTex(r'v = \frac{dx}{dt}')
        self.play(Create(v_def))
        self.next_slide()
        self.play(Transform(v_def, v_def_2, replace_mobject_with_target_in_scene=True))
        self.next_slide()

        x_int_temp = MathTex(r'dx', '=', 'v dt')
        self.play(Transform(v_def_2, x_int_temp, replace_mobject_with_target_in_scene=True))

        self.next_slide()

        x_int_temp_2 = MathTex(r'\int^x_{x_0}', 'dx', '=', r'\int^t_0', 'v dt')
        self.play(TransformMatchingTex(x_int_temp, x_int_temp_2, replace_mobject_with_target_in_scene=True))

        self.next_slide()

        x_int = MathTex('x - x_0', '=', r'\int^t_0', 'v dt')
        self.play(Transform(x_int_temp_2, x_int, replace_mobject_with_target_in_scene=True))

        self.next_slide()

        x_int.generate_target()
        x_int.target.shift(2. * UP + 1.5 * LEFT)

        ax = Axes(x_range=[0, 1, 2], y_range=[0, 1, 2], tips=False)
        labels = ax.get_axis_labels('t', 'v(t)')
        self.play(Create(ax), MoveToTarget(x_int), Create(labels))

        self.next_slide()

        x_square = ax.plot(lambda x: x * x, color=BLUE)
        self.play(Create(x_square))

        self.next_slide()

        rects = VGroup()

        print(rects.submobjects)

        dx_tracker = ValueTracker(0.2)

        text = DecimalNumber(0, num_decimal_places=3, color=BLACK, font_size=72).shift(2.5 * RIGHT + 1.5 * DOWN)

        def rects_updater(m: Mobject):
            # self.remove(*[m1 for m1 in m.submobjects])
            m.remove(*[m1 for m1 in m.submobjects])
            m.submobjects = ax.get_riemann_rectangles(x_square, dx=dx_tracker.get_value(), input_sample_type='center',
                                                      stroke_width=0)
            # self.add(*[m1 for m1 in m.submobjects])

        def text_updater(m: Mobject):
            pts = np.linspace(0, 1, int(1. / dx_tracker.get_value()))
            m.set_value(np.sum(((pts[:-1] + pts[1:]) / 2) ** 2) * dx_tracker.get_value())

        self.add(rects, text)
        text.add_updater(text_updater)
        rects.add_updater(rects_updater)

        self.play(dx_tracker.animate.set_value(0.001), rate_func=rate_functions.ease_in_quad, run_time=4)
        self.wait(0.1)
        # self.next_slide()

        # self.play(*[FadeOut(m) for m in self.mobjects])


class ODEs(Slide):
    def construct(self):
        odes = Tex("Differential Equations", font_size=72)
        self.play(Create(odes))

        title = Title("Differential Equations", include_underline=False)
        self.play(Transform(odes, title))
        self.next_slide()

        fbd_dot = Dot(radius=0.2, fill_color=BLUE)
        self.play(Create(fbd_dot))

        gravity_arrow = Arrow(start=UP, end=DOWN).next_to(fbd_dot, direction=DOWN, buff=0.)

        gravity_tex = MathTex(r'mg', font_size=32).next_to(gravity_arrow, direction=DOWN)

        self.play(GrowArrow(gravity_arrow), Create(gravity_tex))

        fbd_group = VGroup(fbd_dot, gravity_arrow, gravity_tex)
        fbd_group.generate_target()
        fbd_group.target.shift(5 * LEFT)

        self.next_slide()

        self.play(MoveToTarget(fbd_group))

        equations = MathTex(r'ma&=mg \\'
                            r'v&=gt \\'
                            r'y&=\frac{1}{2}gt^2')
        self.play(Create(equations))

        self.next_slide()

        self.play(FadeOut(equations))

        self.next_slide()

        drag_arrow = Arrow(start=DOWN, end=0.5 * UP).next_to(fbd_dot, direction=UP, buff=0.)
        drag_tex = MathTex(r'kv', font_size=32).next_to(drag_arrow, direction=UP)

        self.play(GrowArrow(drag_arrow), Create(drag_tex))

        fbd_group.add(drag_tex, drag_arrow)

        self.next_slide()
        drag_ma = MathTex('m', 'a', '=mg-kv')
        self.play(Create(drag_ma))
        drag_ode = MathTex('m', r'\frac{dv}{dt}', '=mg-kv')

        self.next_slide()
        self.play(TransformMatchingTex(drag_ma, drag_ode))

        self.next_slide()

        drag_equations = MathTex('&', 'm', r'\frac{dv}{dt}', '=mg-kv',
                                 r'\\ &v=\frac{mg}{k}(1-e^{-\frac{kt}{m}}) \\',
                                 r'&y=\frac{mg}{k^2}(kt + (e^{-\frac{kt}{m}} - 1))')

        self.play(TransformMatchingTex(drag_ode, drag_equations))

        self.next_slide()

        drag_equations.generate_target()
        drag_equations.target.shift(5 * LEFT + 1.5 * UP).scale(0.5)

        fbd_group.generate_target()
        fbd_group.target.shift(1.5 * DOWN)

        self.play(MoveToTarget(drag_equations), MoveToTarget(fbd_group))

        ax = Axes(x_range=(0, 1, 10), y_range=(0, 1, 10), x_length=10).shift(2 * RIGHT)

        a_graph = ax.plot(lambda t: np.exp(-t), color=ORANGE, x_range=(0, 1, 0.01))
        v_graph = ax.plot(lambda t: 1 - np.exp(-t), color=BLUE, x_range=(0, 1, 0.01))
        y_graph = ax.plot(lambda t: t - 1 + np.exp(-t), color=GREEN, x_range=(0, 1, 0.01))

        self.next_slide()

        self.play(Create(ax))
        self.play(Create(a_graph), Create(v_graph), Create(y_graph))

        a_label = MathTex('a', font_size=24, color=ORANGE) \
            .next_to(ax.c2p(0.9, np.exp(-0.9)), direction=UP, aligned_edge=RIGHT)
        v_label = MathTex('v', font_size=24, color=BLUE) \
            .next_to(ax.c2p(0.9, 1 - np.exp(-0.9)), direction=UP, aligned_edge=RIGHT)
        y_label = MathTex('y', font_size=24, color=GREEN) \
            .next_to(ax.c2p(0.9, np.exp(-0.9) - .1), direction=DOWN, aligned_edge=RIGHT)

        self.play(Create(a_label), Create(v_label), Create(y_label))


class NewtonRhapson(Slide):
    def construct(self):
        title = Tex('Newton Rhapson Method', font_size=72)
        self.play(Create(title))
        self.wait(0.1)
        self.next_slide()
        title_2 = Title('Newton Rhapson Method', include_underline=False)
        self.play(Transform(title, title_2))
        self.next_slide()
        # create the main equation
        main_eqn = MathTex(r'\cos x', r'=', r'x')
        self.play(Create(main_eqn))

        self.next_slide()

        # morph the equation
        zero_eqn = MathTex(r'\cos x', r'-', 'x', r'=', r'0')
        self.play(TransformMatchingTex(main_eqn, zero_eqn))

        self.next_slide()

        # morph part 2
        f_eqn = MathTex(r'& f(x)', r'=', r'0', r'\\',
                        r'& f(x)', r'=', r'\cos x', r'-', 'x')
        self.play(TransformMatchingTex(zero_eqn, f_eqn))

        self.next_slide()

        # calculate derivative
        f_prime = MathTex(r'& f(x)', r'=', r'0', r'\\',
                          r'& f(x)', r'=', r'\cos x', r'-', 'x', r'\\',
                          r"& f'(x)", r'=', r'-\sin x', r'-1')

        anims = [ReplacementTransform(f_eqn.submobjects[j], f_prime.submobjects[j])
                 for j in range(len(f_eqn.submobjects))]
        anims += [Write(f_prime.submobjects[j]) for j in range(len(f_eqn.submobjects), len(f_prime.submobjects))]
        self.play(*anims)

        self.next_slide()

        # move the equations out of the way

        f_prime.generate_target()
        f_prime.target.scale(0.75)
        f_prime.target.shift(2 * UP + 5 * RIGHT)
        self.play(MoveToTarget(f_prime))

        # plot the actual graph
        axes = Axes(
            x_range=[0, 1.1], y_range=[-0.5, 1.2]
        )
        labels = axes.get_axis_labels(x_label='x', y_label='f(x)')

        def f(x):
            return np.cos(x) - x

        plot = axes.plot(f, color=BLUE, x_range=[0, 1.1])

        self.play(Create(axes), Create(labels))
        self.play(Create(plot))

        def f_prime(x):
            return -np.sin(x) - 1

        guess = 0.01

        point = axes.coords_to_point([[guess, f(guess)]])
        dot = Dot(point=point, fill_color=GREEN)
        self.play(Create(dot))

        for j in range(3):
            def line_func(x):
                return f_prime(guess) * (x - guess) + f(guess)

            new_guess = guess - f(guess) / f_prime(guess)

            line = DashedVMobject(axes.plot(line_func, x_range=[min(new_guess, guess) - 0.1,
                                                                max(new_guess, guess) + 0.1]))
            self.play(Create(line))

            point_on_ax = axes.coords_to_point([[new_guess, 0.]])
            dot_on_ax = Dot(point=point_on_ax, fill_color=GREEN)
            self.play(Create(dot_on_ax))

            point_new_guess = axes.coords_to_point([[new_guess, f(new_guess)]])
            vert_line = DashedLine(point_new_guess, point_on_ax)
            self.play(Create(vert_line))

            new_point = axes.coords_to_point([[new_guess, f(new_guess), 0]])
            new_dot = Dot(point=new_point, fill_color=GREEN)
            self.play(Create(new_dot))

            self.play(FadeOut(line), FadeOut(dot_on_ax), FadeOut(vert_line), FadeOut(dot))

            dot = new_dot
            guess = new_guess

        self.wait(1)


def integrate_euler(f_prime, x_start, x_end, steps, y0):
    dx = (x_end - x_start) / steps

    res = []
    y = y0
    res.append(y)
    for j in range(steps):
        dy_dx = f_prime(y)
        y += dy_dx * dx
        res.append(y)

    return interp.interp1d(np.linspace(x_start, x_end, steps + 1), res), res


class EulersMethod(Slide):
    def construct(self):
        euler = Tex("Euler's Method", font_size=72)
        title = Title("Euler's Method", include_underline=False)
        self.play(Create(euler))
        self.next_slide()
        self.play(Transform(euler, title))
        self.next_slide()

        definition = MathTex(r"\frac{dy}{dx}=y'(x)", font_size=42)
        self.play(Create(definition))
        self.next_slide()
        definition_2 = MathTex(r"d", "y", "=y'(x)", "d", "x")
        self.play(Transform(definition, definition_2))

        d_and_delta_are_the_same = MathTex(r"\Delta", "y", "=y'(x)", r"\Delta", "x")
        self.next_slide()

        self.play(Transform(definition, d_and_delta_are_the_same))
        self.next_slide()
        self.play(FadeOut(definition))

        drag_equation = MathTex(r"m", r"\frac{dv}{dt}", "=", "mg-kv")
        self.play(Create(drag_equation))

        self.next_slide()

        drag_equation_2 = MathTex(r"m", r"d", "v", "=", "(", "mg-kv", ")", "d", "t")
        self.play(TransformMatchingTex(drag_equation, drag_equation_2))

        self.next_slide()

        euler_eqn_temp = MathTex(r"m", r"\Delta", "v", "=", "(", "mg-kv", ")", r"\Delta", "t")
        self.play(Transform(drag_equation_2, euler_eqn_temp))

        self.next_slide()

        euler_eqn = MathTex(r"\Delta", "v", "=", "(", "1-v", ")", r"\Delta", "t")
        self.play(TransformMatchingTex(drag_equation_2, euler_eqn))

        self.next_slide()

        euler_eqn.generate_target()
        euler_eqn.target.scale(0.75).shift(2.5 * UP + 4 * LEFT)

        ax = Axes(x_range=[0., 1.], y_range=[0., 1.])
        labels = ax.get_axis_labels('t', 'v')

        euler_sol = integrate_euler(lambda v: 1 - v, 0, 1, 3, 0)

        graph = ax.plot(euler_sol[0], color=BLUE, x_range=[0, 0])

        self.play(Create(ax), Create(labels), MoveToTarget(euler_eqn), Create(graph))

        pts = np.linspace(0., 1., 4)
        dots = [Dot(ax.c2p(0, 0), color=BLUE)]
        self.play(Create(dots[0]))
        self.next_slide()

        for j in range(3):
            dt_arrow = Arrow(start=ax.c2p(pts[j], euler_sol[1][j]),
                             end=ax.c2p(pts[j + 1], euler_sol[1][j]), color=ORANGE, buff=0)

            dv_arrow = Arrow(start=ax.c2p(pts[j + 1], euler_sol[1][j]),
                             end=ax.c2p(pts[j + 1], euler_sol[1][j + 1]), color=ORANGE, buff=0)

            full_arrow = Arrow(start=ax.c2p(pts[j], euler_sol[1][j]),
                               end=ax.c2p(pts[j + 1], euler_sol[1][j + 1]), color=BLUE, buff=0)

            self.play(Create(dt_arrow))
            self.next_slide()
            self.play(Create(dv_arrow))
            self.next_slide()
            self.play(Create(full_arrow))

            dot = Dot(ax.c2p(pts[j + 1], euler_sol[1][j + 1]), color=BLUE)
            dots.append(dot)
            graph.become(ax.plot(euler_sol[0], color=BLUE, x_range=[0, pts[j + 1]]))
            self.next_slide()
            self.play(Create(dot), FadeOut(dt_arrow), FadeOut(dv_arrow), FadeOut(full_arrow))

            self.next_slide()

        self.play(*[FadeOut(dot) for dot in dots])

        act_graph = DashedVMobject(ax.plot(lambda t: 1 - np.exp(-t), color=GREEN))
        self.play(Create(act_graph))

        self.next_slide()

        for j in range(3, 10):
            self.play(graph.animate.become((ax.plot(integrate_euler(lambda v: 1 - v, 0, 1, j, 0)[0],
                                                    color=BLUE, x_range=[0, 1]))))

        self.next_slide()

        self.play(*[FadeOut(m) for m in self.mobjects])
