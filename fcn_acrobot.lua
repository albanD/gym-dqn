require 'nn'

return function(args)
    local net = nn.Sequential()

    net:add(nn.Linear(args.state_dim, 5))
    net:add(nn.ReLU())
    net:add(nn.Linear(5, args.n_actions))

    return net
end
