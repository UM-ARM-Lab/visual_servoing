class ArmRobot: 
    def get_arm_jacobian(self, side):
        raise NotImplementedError

    def get_jacobian_pinv(self, side):
        raise NotImplementedError

    def psuedoinv_ik_controller(self, side, twist):
        raise NotImplementedError

    def velocity_control(self, side, q_dot):
        raise NotImplementedError
