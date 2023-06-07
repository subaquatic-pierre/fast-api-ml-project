import React from 'react'

const Dashboard = React.lazy(() => import('./views/dashboard/Dashboard'))
const Users = React.lazy(() => import('./views/users/Users'))
const User = React.lazy(() => import('./views/users/User'))
const Cases = React.lazy(() => import('./views/cases/Cases'))
const Case = React.lazy(() => import('./views/cases/Case'))
const ApiKeys = React.lazy(() => import('./views/apikeys/ApiKeys'))

const routes = [
  { path: '/', exact: true, name: 'Home' },
  { path: '/dashboard', name: 'Dashboard', element: Dashboard },
  { path: '/users', name: 'Users', element: Users },
  { path: '/users/:userId', name: 'User', element: User },
  { path: '/cases', name: 'Cases', element: Cases },
  { path: '/cases/:caseId', name: 'Case', element: Case },
  { path: '/api-keys', name: 'API Keys', element: ApiKeys },
]

export default routes
