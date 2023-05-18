from manim import *
import numpy as np


class NewtonsMethod(Scene):
    def construct(self):
        # create the main equation
        main_eqn = MathTex(r'\cos x', r'=', r'x')
        self.play(Create(main_eqn))

        # morph the equation
        zero_eqn = MathTex(r'\cos x', r'-', 'x', r'=', r'0')
        self.play(TransformMatchingTex(main_eqn, zero_eqn))

        # morph part 2
        f_eqn = MathTex(r'& f(x)', r'=', r'0', r'\\',
                        r'& f(x)', r'=', r'\cos x', r'-', 'x')
        self.play(TransformMatchingTex(zero_eqn, f_eqn))

        # calculate derivative
        f_prime = MathTex(r'& f(x)', r'=', r'0', r'\\',
                          r'& f(x)', r'=', r'\cos x', r'-', 'x', r'\\',
                          r"& f'(x)", r'=', r'-\sin x', r'-1')

        anims = [ReplacementTransform(f_eqn.submobjects[j], f_prime.submobjects[j])
                 for j in range(len(f_eqn.submobjects))]
        anims += [Write(f_prime.submobjects[j]) for j in range(len(f_eqn.submobjects), len(f_prime.submobjects))]
        self.play(*anims)

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
