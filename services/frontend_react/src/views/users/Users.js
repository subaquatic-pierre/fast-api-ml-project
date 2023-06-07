import React, { useState, useEffect } from 'react'
import axios from 'axios'

import {
  CTable,
  CTableBody,
  CTableDataCell,
  CTableHead,
  CTableHeaderCell,
  CTableRow,
  CCardBody,
  CCard,
  CCardHeader,
} from '@coreui/react'

import { API_URL } from 'src/const'

const Users = () => {
  const [users, setUsers] = useState([])

  const fetchUsers = async () => {
    try {
      const url = `${API_URL}/user`
      const userRes = await axios.get(url)
      const userResJson = userRes.data
      const users = userResJson.data
      setUsers(users)
    } catch (e) {
      console.log(e)
      setUsers([])
    }
  }

  const handleDeleteUserButtonClick = async (userId) => {
    const url = `${API_URL}/user/${userId}`
    try {
      const res = await axios.delete(url)
      if (res.status === 200) {
        window.location.reload()
      }
    } catch (e) {
      console.log(e)
    }
  }

  useEffect(() => {
    fetchUsers()
  }, [])

  return (
    <>
      <CCard className="mb-4">
        <CCardHeader className="d-flex align-items-center justify-content-between">
          <h5 className="mb-0">Users</h5>
          <div>
            <a className="btn btn-info me-1 text-white" href="/register" type="button">
              <span className="btn-icon ">
                <i className="cil-plus"></i>
              </span>
              New
            </a>
          </div>
        </CCardHeader>
        <CCardBody>
          <CTable align="middle" className="mb-0 " hover responsive>
            <CTableHead color="light">
              <CTableRow className="card-table-row">
                <CTableHeaderCell>User Name</CTableHeaderCell>
                <CTableHeaderCell>Email</CTableHeaderCell>
                <CTableHeaderCell>User Id</CTableHeaderCell>
                <CTableHeaderCell className="actions"></CTableHeaderCell>
              </CTableRow>
            </CTableHead>
            <CTableBody>
              {users.map((user, index) => (
                <CTableRow v-for="user in tableUsers" key={index}>
                  <CTableDataCell>
                    <div>{user.fullName}</div>
                  </CTableDataCell>
                  <CTableDataCell>
                    <div>{user.email}</div>
                  </CTableDataCell>
                  <CTableDataCell>
                    <div>{user.id}</div>
                  </CTableDataCell>
                  <CTableDataCell className="text-end">
                    <div>
                      <a href={`/users/${user.id}`} className="btn btn-primary" type="button">
                        View
                      </a>
                      <button
                        onClick={() => handleDeleteUserButtonClick(user.id)}
                        className="btn btn-danger text-white"
                        type="button"
                      >
                        Delete
                      </button>
                    </div>
                  </CTableDataCell>
                </CTableRow>
              ))}
            </CTableBody>
          </CTable>
        </CCardBody>
      </CCard>
    </>
  )
}

export default Users
