import torch

class BatchVelocityPlanner:
    def __init__(self, max_lin_vel=0.5, max_yaw_vel=1.0, pos_tol=0.05, yaw_tol=0.05):
        """
        Args:
            max_lin_vel: Maximum linear speed (m/s)
            max_yaw_vel: Maximum yaw angular speed (rad/s)
            pos_tol: Position tolerance for arrival detection
            yaw_tol: Yaw tolerance for arrival detection
        """
        self.max_lin_vel = max_lin_vel
        self.max_yaw_vel = max_yaw_vel
        self.pos_tol = pos_tol
        self.yaw_tol = yaw_tol

    def check_arrival(self, pos, yaw, goal_pos, goal_yaw):
        """
        Check whether each environment has reached its target.

        Args:
            pos:      Tensor (N, 2) Current xy positions
            yaw:      Tensor (N,)   Current yaw angles
            goal_pos: Tensor (N, 2) Target xy positions
            goal_yaw: Tensor (N,)   Target yaw angles

        Returns:
            arrived: Tensor (N,) bool
        """
        diff = goal_pos - pos
        dist = torch.norm(diff, dim=1)

        # signed smallest yaw difference
        yaw_error = torch.atan2(
            torch.sin(goal_yaw - yaw),
            torch.cos(goal_yaw - yaw),
        )

        arrived_pos = dist < self.pos_tol
        arrived_yaw = torch.abs(yaw_error) < self.yaw_tol
        return arrived_pos & arrived_yaw

    def compute_cmd(self, pos, yaw, goal_pos, goal_yaw):
        """
        Compute batch velocity commands.

        Returns:
            cmd_vel: Tensor (N, 3)
            arrived: Tensor (N,) bool
        """
        diff = goal_pos - pos
        dist = torch.norm(diff, dim=1)

        # direction unit vector
        direction = diff / (dist.unsqueeze(1) + 1e-8)

        # proportional speed with clamp
        speed = torch.clamp(dist, max=self.max_lin_vel)
        v_xy = direction * speed.unsqueeze(1)

        # yaw tracking
        yaw_error = torch.atan2(
            torch.sin(goal_yaw - yaw),
            torch.cos(goal_yaw - yaw),
        )
        yaw_rate = torch.clamp(yaw_error, -self.max_yaw_vel, self.max_yaw_vel)

        # arrival detection
        arrived = self.check_arrival(pos, yaw, goal_pos, goal_yaw)

        # assemble command
        cmd = torch.zeros((pos.size(0), 3), device=pos.device, dtype=pos.dtype)
        cmd[:, 0:2] = v_xy
        cmd[:, 2] = yaw_rate

        # zero velocity if arrived
        cmd[arrived] = 0.0
        return cmd, arrived
