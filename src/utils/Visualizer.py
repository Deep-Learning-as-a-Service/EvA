
from matplotlib import pyplot


class Visualizer():

    @staticmethod
    def _points_to_two_lists(points):
        xPoints = list(map(lambda point: point[0], points))
        yPoints = list(map(lambda point: point[1], points))
        return xPoints, yPoints

    @staticmethod
    def show_line_from_points(points):
        """
        show_point_plot([(1, 1), (1.1, 3), (7, 8), (0.9, 4)])
        """
        xPoints, yPoints = points_to_two_lists(points)
        pyplot.plot(xPoints, yPoints)
        pyplot.show()

    @staticmethod
    def easy_show_scatter_plot(points):

        # Set the figure size in inches
        pyplot.figure(figsize=(10,6))

        xPoints, yPoints = points_to_two_lists(points)
        pyplot.scatter(xPoints, yPoints, alpha=0.5) # label = "label_name"

        # Set x and y axes labels
        pyplot.xlabel('X Values')
        pyplot.ylabel('Y Values')

        pyplot.title('Scatter Title')
        # pyplot.legend()
        pyplot.show()

    @staticmethod
    def scatter_plot(points):
        assert len(points[0]) == 4, "a point is defined as (x_axis, y_axis, color, alpha)"

        xPoints = list(map(lambda point: point[0], points))
        yPoints = list(map(lambda point: point[1], points))
        colours = list(map(lambda point: point[2], points))
        alphas = list(map(lambda point: point[3], points))
        
        pyplot.figure(figsize=(10,6))

        pyplot.scatter(xPoints, yPoints, c = colours, label = "label_name", alpha=alphas)

        # Set x and y axes labels
        pyplot.xlabel('X Values')
        pyplot.ylabel('Y Values')

        pyplot.title('Scatter Title')
        # pyplot.legend()
        pyplot.show()


# Test
# show_scatter_plot([(5, 5), (5, 5), (6, 7), (1, 2), (5.1, 5), (5, 5.1)])
# show_line_from_points([(1, 1), (1.1, 3), (7, 8), (0.9, 4)])
# show_scatter_plot([(6, 7), (1, 2), (5.1, 5), (5, 5.1)])
# multicolor_scatter_plot([(6, 7, "green"), (1, 2, "red"), (5.1, 5, "red"), (5, 5.1, "blue")])