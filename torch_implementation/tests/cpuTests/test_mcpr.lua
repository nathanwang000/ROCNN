require 'package'
package.path = package.path .. ";../../?.lua" -- add to path

-- this file tests specifics of the network and make sure it works
require 'monteCarloProbRotation'
require 'math'
mcprtest = {}
tester = torch.Tester()

function mcprtest.getRotMatrixTest()
   local A = torch.eye(2)
   tester:assertTensorEq(A,mcpr.get_rot_matrix(0),1e-8)
   tester:assertTensorEq(A,mcpr.get_rot_matrix(math.pi*2),1e-8)
   -- more tests needed TODO!
end

function mcprtest.calc_prob_matrix_test()
   -- TODO
end

function mcprtest.rotate_monte_carlo_test()
   local theta = -math.pi/2
   local filter = torch.Tensor({{0,1,0},{0,2,0},{0,3,0}})
   local target = torch.Tensor({{0,0,0},{3,2,1},{0,0,0}})
   local rot_filter = mcpr.rotate_monte_carlo()(filter,theta)
   -- print('\nrot filter is:') print(rot_filter)
   tester:assertTensorEq(rot_filter,target,0.15)
end

function mcprtest.circshiftTest()
   local A = torch.range(1,10)
   local B = torch.Tensor({10,1,2,3,4,5,6,7,8,9})
   local C = torch.Tensor({3,4,5,6,7,8,9,10,1,2})
   tester:assertTensorEq(mcpr.circshift(A,0,1),A,1e-8,'circshift')
   tester:assertTensorEq(mcpr.circshift(A,-2,1),C,1e-8,'circshift')
   tester:assertTensorEq(mcpr.circshift(A,1,1),B,1e-8,'circshift')
   tester:assertTensorEq(mcpr.circshift(A,8,1),C,1e-8,'circshift')
   tester:assertTensorEq(mcpr.circshift(A,18,1),C,1e-8,'circshift')
end

function mcprtest.circular_rotate_test()
   print('circular rotate test start')
   local filter = torch.range(1,27)
   filter = filter:view(1,3,3,3)
   local theta = math.pi/2
   local rotate_index = 2
   local rotF, prob_matrix = mcpr.rotate_monte_carlo()(filter[{1,2,{},{}}],theta)
   local p = mcpr.circular_rotate(filter,theta,rotate_index,prob_matrix,true)
   tester:assertTensorEq(rotF,p[{1,1,{},{}}],0.1)
end

function mcprtest.circular_rotate_test_dim5()
   local filter = torch.range(1,27)
   filter = filter:view(1,1,3,3,3)
   local theta = math.pi/2
   local rotate_index = 2
   local p = mcpr.circular_rotate(filter,theta,rotate_index,nil,true,false)
   tester:assertTensorEq(mcpr.rotate_monte_carlo()(filter[{1,1,2,{},{}}],theta),p[{1,1,1,{},{}}],0.6)
   print(p[{1,1,1,{},{}}])
   print(filter[{1,1,2,{},{}}])
end

function mcprtest.rotate_test_dim4()
   local filter = torch.range(1,27)
   filter = filter:view(1,3,3,3)
   local theta = math.pi/2
   local p = mcpr.circular_rotate(filter,theta,nil,nil,false,false)
   tester:assertTensorEq(mcpr.rotate_monte_carlo()(filter[{1,1,{},{}}],theta),p[{1,1,{},{}}],0.6)
   print(p[{1,1,{},{}}])
   print(filter[{1,1,{},{}}])
end


tester:add(mcprtest)
tester:run()

