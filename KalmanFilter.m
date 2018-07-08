classdef KalmanFilter < handle
    %KalmanFilter Kalman Filtering object for state estimation.
    
    properties (SetAccess = immutable)
        A double % n x n state transition matrix
        B double % n x m control input matrix
        R double % n x n process noise covariance matrix
        C double % k x n measurement matrix
        Q double % k x k measurement noise covariance matrix
    end
    
    properties (SetAccess = private)
        mu double % n x 1 estimated state vector
        Sigma double % n x n estimated state covariance matrix
    end
    
    properties (SetAccess = immutable, Dependent)
        n double
        m double
        k double
    end
    
    properties(SetAccess = immutable, GetAccess = private, Dependent)
        I double % n x n identity matrix
    end
    
    methods
        function obj = KalmanFilter(A, B, R, C, Q)
            %KalmanFilter constructs a new KalmanFilter object.
            %   A is the n x n state transition matrix.
            %   B is the n x m control inptut matrix.
            %   R is the n x n process noise covariance matrix
            %   C is the k x n measurement matrix
            %   Q is the k x k measurement noise covariance matrix
            
            n = size(A, 1);
            m = size(B, 2);
            k = size(C, 1);

            assert(isequal(size(A), [n n]), "A must be an n x n matrix");
            assert(isequal(size(B), [n m]), "B must be an n x m matrix");
            assert(isequal(size(C), [k n]), "C must be an k x n matrix");
            assert(isequal(size(R), [n n]), "R must be an n x n matrix");
            assert(isequal(size(Q), [k k]), "Q must be an k x k matrix");
            
            obj.A = A;
            obj.B = B;
            obj.R = R;
            obj.C = C;
            obj.Q = Q;

            obj.mu = zeros(n, 1);
            obj.Sigma = 1e-7 .* eye(n); % approximately binary32 epsilon
        end
        
        function [mu, Sigma] = add_observation(obj, u, z)
            %add_observation updates the state of a KalmanFilter.
            %   u is the control vector with length m
            %   z is the measurement vector with length k

            assert(isvector(u) && length(u) == obj.m, ...
                "u must be a vector with length m");
            u = u(:);
            
            assert(isvector(z) && length(z) == obj.k, ...
                "z must be a vector with length k");
            z = z(:);
            
            [mu_bar, Sigma_bar] = obj.predict(u);
            
            [mu, Sigma] = obj.update(z, mu_bar, Sigma_bar);
            
            obj.mu = mu;
            obj.Sigma = Sigma;
        end
        
        function value = get.n(obj)
            %get.n returns the number of state variables
            value = size(obj.A, 1);
        end
        
        function value = get.m(obj)
            %get.m returns the number of control variables
            value = size(obj.B, 2);
        end
        
        function value = get.k(obj)
            %get.k returns the number of observation variables
            value = size(obj.C, 1);
        end
        
        function I = get.I(obj)
            I = eye(obj.n);
        end
    end
    
    methods (Access = private)
        function [mu, Sigma] = predict(obj, u)
            mu = obj.A * obj.mu * obj.B * u;
            Sigma = obj.A * obj.Sigma * obj.A.' + obj.R;
        end
        
        function [mu, Sigma] = update(obj, z, mu_bar, Sigma_bar)
            K = Sigma_bar * obj.C.' * ...
                inv(obj.C * Sigma_bat * obj.C.' + obj.Q);

            mu = mu_bar + K * (z - obj.C * mu_bar);
            Sigma = (obj.I - K * obj.C) * Sigma_bar;
        end
    end
end

